import torch
import torch.nn.functional as F
from functools import partial
import collections
import torch.nn as nn
import numpy as np

#import opt_einsum as oe

from pyaml_env import parse_config
config = parse_config("./src/config.yaml")

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_module(model, module_name, model_layer_name, layer=None):
    parsed_module_name = module_name.split('.')
    tmp_module = model
    if layer != None:
        parsed_layer_name = model_layer_name.split('.')
        # Loop to find layers module
        for sub_module in parsed_layer_name:
            tmp_module = getattr(tmp_module, sub_module)
        # Select specific layer
        tmp_module = tmp_module[layer]
    # Loop over layer module to find module_name
    for sub_module in parsed_module_name:
        tmp_module = getattr(tmp_module, sub_module)

    return tmp_module

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()

        self.model = model

        self.modules_config = config['models'][model.config.model_type]

        self.num_attention_heads = self.model.config.num_attention_heads
        self.attention_head_size = int(self.model.config.hidden_size / self.model.config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

    def save_activation(self,name, mod, inp, out):
        self.func_inputs[name].append(inp)
        self.func_outputs[name].append(out)

    def clean_hooks(self):
        for k, v in self.handles.items():
            self.handles[k].remove()
    @torch.no_grad()
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def get_modules_model(self, layer):
        '''Gets Transformer modules (weights).'''

        model_layer_name = self.modules_config['layer']

        dense = get_module(self.model, self.modules_config['dense'], model_layer_name, layer)
        fc1 = get_module(self.model, self.modules_config['fc1'], model_layer_name, layer)
        fc2 = get_module(self.model, self.modules_config['fc2'], model_layer_name, layer)
        ln1 = get_module(self.model, self.modules_config['ln1'], model_layer_name, layer)
        ln2 = get_module(self.model, self.modules_config['ln2'], model_layer_name, layer)
        values = get_module(self.model, self.modules_config['values'], model_layer_name, layer)

        return {'dense': dense,
                'fc1': fc1,
                'fc2': fc2,
                'ln1': ln1,
                'ln2': ln2,
                'values': values
                }
    
    def get_values_weights(self, values_module):
        '''Extract the weights and bias of the values linear projection in the attention mechanism.
        We need:
            W^l_v in shape: [dim, num_heads, dim_head]
            b^l_v in shape: [num_heads, dim_head]
        '''

        if self.model.config.model_type == 'gpt2':
            w_v = values_module.weight[:,-self.all_head_size:]
            w_v = w_v.view(-1,self.num_attention_heads,self.attention_head_size)
            # b^l_v -> num_heads, dim_head
            b_v = values_module.bias[-self.all_head_size:].view(self.num_attention_heads,-1)

        elif self.model.config.model_type == 'opt':
            w_v = values_module.weight.transpose(0,1)
            w_v = w_v.view(-1,self.num_attention_heads,self.attention_head_size)
            # b^l_v -> num_heads, dim_head
            b_v = values_module.bias[-self.all_head_size:].view(self.num_attention_heads,-1)
        elif self.model.config.model_type == 'bloom':
            # BLOOM has a weird way of computing the values
            # https://github.com/huggingface/transformers/blob/cf11493dce0a1d22446efe0d6c4ade02fd928e50/src/transformers/models/bloom/modeling_bloom.py#LL238C9-L238C21
            w_v_big = values_module.weight.transpose(0,1)
            split_w_v_big = w_v_big.view(-1, self.num_attention_heads, 3, self.attention_head_size)
            w_v = split_w_v_big[:,:,2,:].reshape(self.all_head_size,self.all_head_size)
            w_v = w_v.view(-1,self.num_attention_heads,self.attention_head_size)
            # b_v -> num_heads, dim_head
            b_v = values_module.bias.view(self.num_attention_heads, 3, self.attention_head_size)[:,2,:]
        else:
            w_v = values_module.weight.transpose(0,1)
            w_v = w_v.view(-1,self.num_attention_heads,self.attention_head_size)
            # b^l_v -> num_heads, dim_head
            b_v = values_module.bias[-self.all_head_size:].view(self.num_attention_heads,-1)
            
        return w_v, b_v
    
    def get_out_proj_weights(self, out_proj_module):
        '''Extract the weights and bias of the output linear projection in the attention mechanism.
        We need:
            W^l_o in shape: [dim, num_heads, dim_head]
            b^l_o in shape: [dim]
        '''
        if self.model.config.model_type == 'gpt2':
            dense = out_proj_module.weight.transpose(0,1)
        else:
            dense = out_proj_module.weight
        
        w_o = dense.view(self.all_head_size, self.num_attention_heads, self.attention_head_size)
        b_o = out_proj_module.bias.detach()

        return w_o, b_o

    def l_transform(self, x, w_ln, ln_eps, pre_ln_states):
        '''Computes mean and performs hadamard product with ln weight (w_ln) as a linear transformation.
            Also divides by the SD of the input vector.
            out = number of positions in forwards-pass (number of residuals)
            Input:
                x (tensor): tensor to which apply l_transform on its out dimension
                w_ln (tensor): weights of layer norm
                ln_eps (float): epsilon of layer norm
                pre_ln_states (tensor) -> [batch_size, out, dim]: the states (or values) of the tensor just
                                before LN is applied
            Output:
                output (tensor) -> [batch_size, out, int_dim, dim]
        '''

        # Create square matrix with γ parameters in diagonal
        # ln_param_transf -> [dim, dim]
        ln_param_transf = torch.diag(w_ln)
        ln_mean_transf = torch.eye(w_ln.size(0)).to(w_ln.device) - \
            1 / w_ln.size(0) * torch.ones_like(ln_param_transf).to(w_ln.device)

        # Compute variance of pre final layernorm states (variance computed individual for each out position)
        # var_pre_ln_states -> [out]
        var_pre_ln_states = torch.var(pre_ln_states, unbiased=False, dim=-1)

        # Add epsilon value to each position [out]
        # Square root (element-wise) -> [out]
        # ln_std_coef -> [out]
        ln_std_coef = 1/torch.sqrt(var_pre_ln_states +  ln_eps)

        # Compute main operation
        # out [batch_size, out, int_dim, dim] , int_dim is the intermediate dimension of MLP
        output = torch.einsum(
            '... e , e f , f g -> ... g',
            x,
            ln_mean_transf,
            ln_param_transf
        )
        # Move ln_std_coef values to first dim to multiply elementwise with out dimension of out
        # ln_std_coef -> [out, 1, 1]
        ln_std_coef = ln_std_coef.view(-1,1,1)
        output = output*ln_std_coef
        
        return output
    
    def run_final_ln(self, tensor_input, pre_lnf_states, lnf):
        if lnf != None:
            lnf_weight, lnf_eps, _ = lnf.weight.data.detach(), lnf.eps, lnf.bias.detach()
            return self.l_transform(tensor_input, lnf_weight, lnf_eps, pre_lnf_states)
        else:
            # If no lnf, return initial tensor as is
            return tensor_input
    
    @torch.no_grad()
    def get_logit_contributions(self, hidden_states, attentions, token, full_vocab=False, output_pos=-1):
        '''Obtains the ∆logit to token by each Transformer component.
            Args:
                hidden_states_model (list[torch.tensor]): intermediate token representations at each layer (inputs to self-attention). [batch, seq_length, all_head_size]
                attentions (list[(torch.tensor)]): attention weights calculated in self-attention. [batch, num_heads, seq_length, seq_length]
                token (list[torch.tensor]): token indexes to get explanations from. If full_vocab = True, the entire vocabulary is used.
                full_vocab (boolean): Get explanations w.r.t the whole output vocabulary or just w.r.t token. full_vocab=True is way inefficient.
                output_pos (int): position of token w.r.t we want explanations (-1 for Language Modeling)
        '''
        logits_modules = {}
        logits_modules['mlp_logit_layers'] = []
        logits_modules['b_o_logits_layers'] = []

        # Get number of layers in model
        try:
            num_layers = self.model.config.n_layers
        except:
            num_layers = self.model.config.num_hidden_layers

        # Cached activations of every layer in the model after the forward-pass
        func_outputs = self.func_outputs
        func_inputs = self.func_inputs

        # Last layer info
        modules_dict = self.get_modules_model(num_layers-1)
        model_info_dict = self.modules_config
        model_layer_name = model_info_dict['layer']

        # Last residual state = output MLP (last_fc2_out) + residual before MLP (last_pre_ln2_states)
        # TODO. This doesn't hold for post-ln models!
        last_pre_ln2_states = func_inputs[model_layer_name + '.' + str(num_layers-1) + '.' + self.modules_config['ln2']][0][0].squeeze()
        last_fc2_out = func_outputs[model_layer_name + '.' + str(num_layers-1) + '.' + self.modules_config['fc2']][0].squeeze()

        # Input of Final Layernorm
        # Will use this to project through the unembedding matrix
        pre_lnf_states = last_pre_ln2_states + last_fc2_out


        list_logits_l_lin_x_j_heads = []
        list_logits_l_aff_x_j_heads = []
        list_logits_l_lin_x_j = []
        list_logits_l_aff_x_j = []
        model_importance_list = []
        attn_res_output_list = []
        
        lnf = get_module(self.model, self.modules_config['lnf'], model_layer_name)

        # Get columns of Unembedding matrix
        # If full_vocab is used, the entire unembedding matrix is used
        embed = get_module(self.model, self.modules_config['unembed'], model_layer_name)
        embed_matrix = embed.weight.detach()
        if full_vocab == False:
            embed_matrix = embed_matrix[token]
        else:
            embed_matrix = embed_matrix
            
        for layer in range(0,num_layers):
            hidden_states_layer = hidden_states[layer].detach()
            modules_dict = self.get_modules_model(layer)
            model_layer_name = self.modules_config['layer']

            if layer == 0:                  
                # ∆logit^l x^0
                # Inital value (layer 0) of the residual at output_pos -> [dim]
                initial_residual = hidden_states_layer.squeeze()[output_pos]

                lnf_initial_residual = self.run_final_ln(initial_residual, pre_lnf_states[output_pos], lnf).squeeze()
                init_logit = torch.einsum('vd,d->v', embed_matrix, lnf_initial_residual)
                logits_modules['init_logit'] = init_logit

                # ∆logit final layernorm bias
                if lnf != None:
                    # Get final layernorm (if exists)
                    lnf_bias = lnf.bias.detach()
                    lnf_bias_logit = torch.einsum('vd,d->v', embed_matrix, lnf_bias)
                    logits_modules['lnf_bias'] = lnf_bias_logit

            # ∆logit^l MLP
            fc2_out = func_outputs[model_layer_name + '.' + str(layer) + '.' + self.modules_config['fc2']][0].squeeze()
            # Apply final layernorm
            l_fc2_out = self.run_final_ln(fc2_out.unsqueeze(-2), pre_lnf_states, lnf)
            mlp_logit = torch.einsum('vd,td->tv', embed_matrix, l_fc2_out.squeeze())
            logits_modules['mlp_logit_layers'].append(mlp_logit[output_pos])

            ln1 = get_module(self.model, self.modules_config['ln1'], model_layer_name, layer)
            ln1_eps = ln1.eps
            w_ln1 = ln1.weight.data.detach()
            b_ln1 = ln1.bias
            a_mat = attentions[layer].squeeze()

            # Extract W^l_V and b_v
            w_v, b_v = self.get_values_weights(modules_dict['values'])
            # Extract W^l_O and b_o
            w_o, b_o = self.get_out_proj_weights(modules_dict['dense'])

            # ∆logit^l b^l_o (Bias linear transformation attn out_proj)
            # b_o = modules_dict['dense'].bias.detach()
            lnf_b_o = self.run_final_ln(b_o, pre_lnf_states, lnf)
            # lnf_b_o_e -> [seq_len, vocab_size]
            lnf_b_o_e = torch.einsum('vd,tsd->tv', embed_matrix, lnf_b_o)
            logits_modules['b_o_logits_layers'].append(lnf_b_o_e[output_pos])

            # Compute biases term A⋅θ
            b_ln1_w_v = torch.einsum('d,dhz->hz', b_ln1, w_v) + b_v
            gamma_l_h = torch.einsum('hz,dhz->hd', b_ln1_w_v, w_o)
            biases_term_attn_heads = torch.einsum('hts,hd->htsd', a_mat, gamma_l_h)
            # Sum over heads
            # biases_term_src -> [t, s, d]
            biases_term_src = biases_term_attn_heads.sum(0)
            # Sum over inputs (j paper) and b_o
            # biases_term -> [t, d]
            biases_term = biases_term_src.sum(-2) + b_o

            # layer_input -> [s, 1, d]
            layer_input = self.func_inputs[model_layer_name + '.' + str(layer) + '.' + self.modules_config['ln1']][0][0].squeeze().unsqueeze(1)
            # l_layer_input -> [s, d]
            l_layer_input = self.l_transform(layer_input, w_ln1, ln1_eps, layer_input).squeeze()# + b_ln1
            # v_j_heads -> [num_heads, t, head_dim]
            v_j_heads = torch.einsum('sd,dhz->hsz', l_layer_input, w_v).squeeze()# + b_v.unsqueeze(1)
            # lin_x_j_w_o_heads -> [num_heads, t, dim]
            lin_x_j_w_o_heads = torch.einsum('hsz,dhz->hsd', v_j_heads, w_o)

            # Head-level transformed vectors
            # Linear component
            lin_x_j_heads = torch.einsum('hsd,hts->htsd', lin_x_j_w_o_heads, a_mat)
            l_lin_x_j_heads = self.run_final_ln(lin_x_j_heads, pre_lnf_states, lnf)
            logits_l_lin_x_j_heads = torch.einsum('htsd,vd->htvs', l_lin_x_j_heads, embed_matrix).detach().cpu()
            list_logits_l_lin_x_j_heads.append(logits_l_lin_x_j_heads.unsqueeze(0))
            
            # Affine component
            aff_x_j_heads = biases_term_attn_heads + lin_x_j_heads
            l_aff_x_j_heads = self.run_final_ln(aff_x_j_heads, pre_lnf_states, lnf)
            logits_l_aff_x_j_heads = torch.einsum('htsd,vd->htvs', l_aff_x_j_heads, embed_matrix).detach().cpu()
            list_logits_l_aff_x_j_heads.append(logits_l_aff_x_j_heads.unsqueeze(0))

            # Layer-level transformed vectors
            # Linear component
            lin_x_j = lin_x_j_heads.sum(0) # sum over heads lin_x_j -> [t, s, d]
            l_lin_x_j = self.run_final_ln(lin_x_j, pre_lnf_states, lnf)
            logits_l_lin_x_j = torch.einsum('tsd,vd->tvs', l_lin_x_j, embed_matrix).detach().cpu()
            list_logits_l_lin_x_j.append(logits_l_lin_x_j.unsqueeze(0))

            # Affine component T(x_j) = linear part + biases terms
            aff_x_j = biases_term_src + lin_x_j
            l_aff_x_j = self.run_final_ln(aff_x_j, pre_lnf_states, lnf)
            logits_l_aff_x_j = torch.einsum('tsd,vd->tvs', l_aff_x_j, embed_matrix).detach().cpu()
            list_logits_l_aff_x_j.append(logits_l_aff_x_j.unsqueeze(0))

            # Make residual matrix -> [seq_length, seq_length, all_head_size]
            hidden_shape = hidden_states_layer.squeeze().size()
            device = hidden_states_layer.device
            residual = torch.einsum('sk,sd->skd', torch.eye(hidden_shape[0],dtype=hidden_states_layer.dtype).to(device), hidden_states_layer.squeeze())
            res_and_aff_x_j = aff_x_j + residual
            # Get output attention + res -> Sum over inputs (j) and add b^l_o
            attn_res_output = res_and_aff_x_j.sum(1) + b_o

            # Get actual output attention + res (input MLP layernorm) from forward-pass activations
            pre_ln2_states = func_inputs[model_layer_name + '.' + str(layer) + '.' + self.modules_config['ln2']][0][0]
            real_attn_res_output = pre_ln2_states
            # Check if our attn_output matches real_attn_output
            assert torch.dist(attn_res_output, real_attn_res_output).item() < 1e-3 * real_attn_res_output.numel()

            # TODO: add as argument in function
            importance_matrix = -F.pairwise_distance(res_and_aff_x_j, attn_res_output.unsqueeze(1),p=1)

            model_importance_list.append(torch.squeeze(importance_matrix).cpu().detach())
            attn_res_output_list.append(torch.squeeze(attn_res_output).cpu().detach())

        
        tensor_logits_l_lin_x_j_heads = torch.vstack(list_logits_l_lin_x_j_heads)
        tensor_logits_l_aff_x_j_heads = torch.vstack(list_logits_l_aff_x_j_heads)

        tensor_logits_l_lin_x_j = torch.vstack(list_logits_l_lin_x_j)
        tensor_logits_l_aff_x_j = torch.vstack(list_logits_l_aff_x_j)

        layerwise_contribution_layers = torch.stack(model_importance_list)
        attn_res_outputs_layers = torch.stack(attn_res_output_list)

        logit_trans_vect_dict = {
            'lin_x_j_h': tensor_logits_l_lin_x_j_heads, # Logits Linear part of head-wise decomposition
            'aff_x_j_h': tensor_logits_l_aff_x_j_heads, # Logits Affine part of head-wise decomposition
            'lin_x_j' : tensor_logits_l_lin_x_j, # Logits Linear part of layer-wise decomposition
            'aff_x_j': tensor_logits_l_aff_x_j # Logits Affine part of head-wise decomposition
            }
        
        layer_alti_data = {
            'layerwise_contributions': layerwise_contribution_layers, # Layer-wise contributions obtained for ALTI
            'attn_res_outputs': attn_res_outputs_layers # Output attention + residual (later used for ALTI)
            }

        return logit_trans_vect_dict, logits_modules, layer_alti_data 

    def get_prediction(self, input_model):
        with torch.no_grad():
            output = self.model(input_model, output_hidden_states=True, output_attentions=True)
            prediction_scores = output['logits']

            return prediction_scores

    def __call__(self,input_model):
        with torch.no_grad():
            self.handles = {}
            for name, module in self.model.named_modules():
                self.handles[name] = module.register_forward_hook(partial(self.save_activation, name))

            self.func_outputs = collections.defaultdict(list)
            self.func_inputs = collections.defaultdict(list)

            output = self.model(**input_model, output_hidden_states=True, output_attentions=True)
            prediction_scores = output['logits'].detach()
            hidden_states = output['hidden_states']
            attentions = output['attentions']
            
            # Clean forward_hooks dictionaries
            self.clean_hooks()
            return prediction_scores, hidden_states, attentions

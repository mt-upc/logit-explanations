#!/bin/bash
#SBATCH -p veu             # Partition to submit to
#SBATCH --mem=20G      # Max CPU Memory
#SBATCH --job-name=ioi_sva
#SBATCH --gres=gpu:1
#SBATCH --output=/home/usuaris/veu/javier.ferrando/logs_jupyter/%j.out

set -ex

export LC_ALL=en_US.UTF-8
export PATH=~/anaconda3/bin:$PATH

export PYTHONUNBUFFERED=TRUE


source activate alti

declare -a blimp1=("anaphor_gender_agreement" "anaphor_number_agreement" "animate_subject_passive")
declare -a blimp2=("determiner_noun_agreement_1" "determiner_noun_agreement_irregular_1" "determiner_noun_agreement_with_adjective_1" "determiner_noun_agreement_with_adj_irregular_1" "npi_present_1" "distractor_agreement_relational_noun")
declare -a ioi_sva=("ioi" "sva_1" "sva_2" "sva_3" "sva_4")

for sub_dataset in "${ioi_sva[@]}"
do
    for method in grad erasure
    do
        python extract_explanations.py --name_path gpt2  \
                                        --dataset $sub_dataset \
                                        --explanation_type $method
    done
done

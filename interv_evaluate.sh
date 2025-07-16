perspectives="protagonist"
beliefs="TrueBelief FalseBelief"
directions="Coef"
# Ks="16 32 64"
Ks="14 28 56"
alphas="-10 5 0 5 10"

for K in $Ks
do
    for alpha in $alphas
    do
        for direction in $directions
        do
            for perspective in $perspectives
            do
                for belief in $beliefs
                do
                    CUDA_VISIBLE_DEVICES="0" python evaluate.py \
                        --model RepBelief_models/$1 \
                        --dataset GridToM \
                        --annotation GridToM/info.json \
                        --output_dir Vib_results \
                        --image_process_mode Default \
                        --seed 42 \
                        --temperature 0 \
                        --top_p 0.7 \
                        --max_new_tokens 200 \
                        --indice_num -1 \
                        --belief $belief \
                        --perspective $perspective \
                        --intervene True \
                        --alpha $alpha \
                        --K $K \
                        --direction $direction 
                done
            done
        done
    done
done
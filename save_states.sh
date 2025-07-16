beliefs="TrueBelief FalseBelief"

for belief in $beliefs
do
CUDA_VISIBLE_DEVICES="0" python save_states.py \
    --model RepBelief_models/$1 \
    --dataset GridToM/ \
    --annotation GridToM/info.json \
    --output_dir Vib_results \
    --image_process_mode Default \
    --seed 42 \
    --temperature 0 \
    --top_p 0.7 \
    --max_new_tokens 512 \
    --indice_num -1 \
    --belief $belief
done
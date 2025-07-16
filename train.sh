perspectives="protagonist oracle"

for perspective in $perspectives
do
    CUDA_VISIBLE_DEVICES="0" python train.py \
        --annotation GridToM/info.json \
        --output_dir first_order/Vib_results \
        --seed 42 \
        --perspective $perspective
done
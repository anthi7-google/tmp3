export PYTHONPATH=./

declare -A dataset_to_window_map

#"Weather" "ETTh1" "ExchangeRate" "Electricity" "Traffic"
#"96" "168" "336" "720"
#"iTransformer" "SCINet" "DLinear" "PatchTST" "FITS"

models=("DLinear")
adds=("DSD")
datasets=("Electricity")
pred_lens=("96" "168" "336" "720")
device="cuda:0"
windows=96
for model in "${models[@]}"
    do
    for add in "${adds[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            for pred_len in "${pred_lens[@]}"
            do
                CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/add_experiments/$model.py --dataset_type="$dataset" --add_type="$add" --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows=$windows --epochs=100 --lv=2 runs --seeds='[2024]'
            done
        done
    done
done

echo "All runs completed."


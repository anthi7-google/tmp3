export PYTHONPATH=./

declare -A dataset_to_window_map

# "ETTh1" "Weather" "ExchangeRate" "Electricity" "Traffic"
# "SCINet" "FITS"
# "No"
# "96" "168" "336" "720"

models=("FITS")
adds=("No")
datasets=("ETTh1" "Weather" "ExchangeRate" "Electricity" "Traffic")
pred_lens=("96" "168" "336" "720")
device="cuda:0"
windows=96    
# ./run.sh "dlinear" "No" "ExchangeRate" "24"  "cuda:0"
for model in "${models[@]}"
do
    for add in "${adds[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            # windows=${dataset_to_window_map[$dataset]}
            for pred_len in "${pred_lens[@]}"
            do
                echo "Running with dataset = $dataset and pred_len = $pred_len , window = $windows"
                CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/add_experiments/$model.py   --dataset_type="$dataset" --add_type="$add" --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows=$windows --epochs=100 runs --seeds='[2024]'
            done
        done
    done
done
echo "All runs completed."

# 1. 数据根目录：指到 unsupervised_anomaly_detection 这一层
datapath=/root/autodl-tmp/InsPlad/SimpleNet/data/unsupervised_anomaly_detection

# 2. 自动抓取全部子类名（17 个目录）
datasets=($(ls "$datapath"))

# 3. 拼出 -d 参数数组
dataset_flags=($(for dataset in "${datasets[@]}"; do echo "-d" "$dataset"; done))

# 4. 调用 SimpleNet 主程序
python3 main.py \
  --gpu 0 \
  --seed 0 \
  --log_group simplenet_insplad \
  --log_project InsPLAD_Results \
  --results_path results \
  --run_name run \
  net \
  -b wideresnet50 \
  -le layer2 \
  -le layer3 \
  --pretrain_embed_dimension 1536 \
  --target_embed_dimension 1536 \
  --patchsize 3 \
  --meta_epochs 40 \
  --embedding_size 256 \
  --gan_epochs 4 \
  --noise_std 0.015 \
  --dsc_hidden 1024 \
  --dsc_layers 2 \
  --dsc_margin .5 \
  --pre_proj 1 \
  dataset \
  --batch_size 8 \
  --resize 329 \
  --imagesize 288 \
  "${dataset_flags[@]}" mvtec "$datapath"
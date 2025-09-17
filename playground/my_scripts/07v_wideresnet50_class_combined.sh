datapath=/kaggle/working/insplad
datasets=('class_combined')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python3 main.py \
--gpu 0 \
--seed 0 \
--log_group simplenet_insplad \
--log_project INSPLAD_Results \
--results_path results \
--run_name 07v_wideresnet50_class_combined \
net \
-b wideresnet50 \
-le layer2 \
-le layer3 \
--pretrain_embed_dimension 1536 \
--target_embed_dimension 1536 \
--patchsize 3 \
--meta_epochs 10 \
--embedding_size 256 \
--gan_epochs 2 \
--noise_std 0.015 \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin .5 \
--pre_proj 1 \
--save_frequency 2 \
--use_depth \
--fg_noise_std 0.05 \
--bg_noise_std 0.01 \
--depth_threshold 0.5 \
dataset \
--batch_size 16 \
--resize 329 \
--num_workers 4 \
--imagesize 288 "${dataset_flags[@]}" mvtec $datapath

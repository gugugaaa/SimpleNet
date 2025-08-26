datapath=/kaggle/working/insplad
datasets=('polymer-insulator-lower-shackle')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python3 main.py \
--gpu 0 \
--seed 0 \
--log_group simplenet_insplad \
--log_project INSPLAD_Results \
--results_path results \
--run_name 06t_wideresnet50_class_new_1 \
net \
-b wideresnet50 \
-le layer2 \
-le layer3 \
--pretrain_embed_dimension 1536 \
--target_embed_dimension 1536 \
--patchsize 3 \
--meta_epochs 25 \
--embedding_size 256 \
--gan_epochs 5 \
--noise_std 0.02 \
--dsc_hidden 2048 \
--dsc_layers 3 \
--dsc_margin .3 \
--dsc_lr 0.0001 \
--pre_proj 2 \
--save_frequency 2 \
dataset \
--batch_size 16 \
--resize 329 \
--num_workers 4 \
--imagesize 288 "${dataset_flags[@]}" mvtec $datapath

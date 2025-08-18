datapath=/path/to/insplad
datasets=('polymer-insulator-lower-shackle')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python3 main.py \
--gpu 0 \
--seed 0 \
--log_group simplenet_insplad \
--log_project INSPLAD_Results \
--results_path results \
--run_name 02t_resnet18_c10_noise \
net \
-b resnet18 \
-le layer2 \
-le layer3 \
--pretrain_embed_dimension 1536 \
--target_embed_dimension 1536 \
--patchsize 3 \
--meta_epochs 15 \
--embedding_size 256 \
--gan_epochs 3 \
--noise_std 0.02 \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin .5 \
--dsc_lr 0.00005 \
--pre_proj 1 \
dataset \
--batch_size 16 \
--resize 329 \
--num_workers 4 \
--imagesize 288 "${dataset_flags[@]}" mvtec $datapath
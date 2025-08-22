# 01v: 表示这是第一次运行，处于验证模式，区别于训练/预测模式
# resnet18: 使用的网络架构
# c10: 表示只测试了第十个子文件夹 (class 10)
# proj: 表示启用了常规的投影网络 (--pre_proj 1)
# gauss: 表示使用了高斯噪声 (--noise_std 0.015)

datapath=/path/to/insplad
datasets=('polymer-insulator-lower-shackle')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python3 main.py \
--gpu 0 \
--seed 0 \
--log_group simplenet_insplad \
--log_project INSPLAD_Results \
--results_path results \
--run_name 01v_resnet18_class_10 \
net \
-b resnet18 \
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
dataset \
--batch_size 16 \
--resize 329 \
--num_workers 4 \
--imagesize 288 "${dataset_flags[@]}" mvtec $datapath

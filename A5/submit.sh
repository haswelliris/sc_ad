source haienv 202111
# hfai + haiscale
# 2 node 16 GPUs
hfai python train_vgg16.py -- -n 4 -i ubuntu2004-cu113 -g jd_a100#A --name vgg_hfai_nodes_2
# 4 node 32 GPUs
hfai python train_vgg16.py -- -n 4 -i ubuntu2004-cu113 -g jd_a100#A --name vgg_hfai_nodes_4
# 8 node 64 GPUs
hfai python train_vgg16.py -- -n 8 -i ubuntu2004-cu113 -g jd_a100#A --name vgg_hfai_nodes_8
# 16 node 128 GPUs
hfai python train_vgg16.py -- -n 16 -i ubuntu2004-cu113 -g jd_a100#A --name vgg_hfai_nodes_16
# Torch + FSDP
# 2 node 16 GPUs
hfai python train_vgg16.py -- -n 4 -i ubuntu2004-cu113 -g jd_a100#A --name vgg_hfai_nodes_2
# 4 node 32 GPUs
hfai python train_vgg16_torch.py -- -n 4 -i ubuntu2004-cu113 -g jd_a100#A --name vgg_torch_nodes_4
# 8 node 64 GPUs
hfai python train_vgg16_torch.py -- -n 8 -i ubuntu2004-cu113 -g jd_a100#A --name vgg_torch_nodes_8
# 16 node 128 GPUs
hfai python train_vgg16_torch.py -- -n 16 -i ubuntu2004-cu113 -g jd_a100#A --name vgg_torch_nodes_16
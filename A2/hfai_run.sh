source haienv 202207
# submit for PCIe A100
hfai python resnet.py -- -n 1 -i ubuntu2004-cu113 -g jd_a100 --name resnet
# submit for DGX A100
hfai python resnet.py -- -n 1 -i ubuntu2004-cu113 -g jd_a100#DGX --name resnet_dgx
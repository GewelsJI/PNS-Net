import torch
path ='/media/deep/2021-MICCAI-Polyp/Code/MICCAI2021-Experiments/PNS-Net/snapshot/PNS-Net/backbone.pth'

stae = torch.load(path)
new = {k.replace('combine','decoder').replace('low_dilation_conv_group','Low_RFB').replace('dilation_conv_group','High_RFB'):v for k, v in stae.items()}
torch.save(new,'/media/deep/2021-MICCAI-Polyp/Code/MICCAI2021-Experiments/PNS-Net/snapshot/PNS-Net/backbone_new.pth')
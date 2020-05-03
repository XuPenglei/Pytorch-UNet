from unet import UNetWithBackbone,ResNetUnet
import torch
if __name__ =='__main__':
    x = torch.zeros((2,3, 512, 512))
    unet = ResNetUnet(1)
    r = unet(x)
    print(r)

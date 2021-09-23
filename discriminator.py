"""
reference: https://youtu.be/4LktBHGCNfw Aladdin Persson
"""
import torch
import torch.nn as nn
from torch.nn.modules.container import T

# Conv Block
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode='reflect'), # Reflection padding was used to reduce artifacts(artifacts: the pixels that look like artificially generated, I assume)
            nn.InstanceNorm2d(out_channels), # exception
            nn.LeakyReLU(0.2),  # both G, D will use LeakyReLU
        )
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.iniitial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode='reflect',
            ),
            nn.LeakyReLU(0.2), # no InstanceNorm
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature==features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')) # output value is single value between 0 or 1 that indicates if it's a real or fake image.
        self.model = nn.Sequential(*layers) # unpacking layers
    
    def forward(self, x):
        x = self.iniitial(x)
        return torch.sigmoid(self.model(x)) # return between 0 or 1 using torch.sigmoid

def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand((10,3,256, 256)).to(device)
    model = Discriminator(in_channels=3).to(device)
    pred = model(x)
    print(pred.shape)

if __name__=="__main__":
    test()

    """
    torch.Size([10, 1, 30, 30])
    Each value in  30 by 30 grid sees a 70 by 70 patch in the original image.
    ------------------------------------------------------------------------------
    paper says:
    'For the discriminator networks we use 70 × 70 PatchGANs, 
    which aim to classify whether 70 × 70 overlapping image patches are real or fake. 
    Such a patch-level discriminator architecture has fewer parameters than a full-image discriminator 
    and can work on arbitrarilysized images in a fully convolutional fashion.'
    """

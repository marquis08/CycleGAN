import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs), # TODO:about ConvTranspose
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()

        )
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),                             # activation True
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),     # activation False as this is residual
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        """
        paper says:

        We use 6 blocks for 128 × 128 images and 
        9 blocks for 256×256 and higher-resolution training images.
        """
        # conv block without instancenorm
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.ReLU(inplace=True),

        )
        # 2 conv blocks(dowm sample)
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),

            ]
        )
        # 
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features*2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
            ]
        )
        # convert to RGB
        self.last = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
    
    def forward(self, x):
        x = self.initial(x)
        
        for layer in self.down_blocks:
            x = layer(x)

        x = self.residual_blocks(x)

        for layer in self.up_blocks:
            x = layer(x)
        
        assert(list(x.shape[2:]) == [256, 256])
        
        return torch.tanh(self.last(x)) # range from -1 to 1


def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_channels = 3
    img_size = 256
    x = torch.rand((10,img_channels, img_size, img_size)).to(device)
    model = Generator(img_channels=3, num_residuals=9).to(device)
    print(model(x).shape)

if __name__=="__main__":
    test()
    """
    output should be like this
    torch.Size([10, 3, 256, 256])
    """
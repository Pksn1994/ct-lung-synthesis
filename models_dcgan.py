import torch.nn as nn

nz = 100

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        ]
        for incoming, outgoing in [(512, 256), (256, 128), (128, 64)]:
            layers += [
                nn.ConvTranspose2d(incoming, outgoing, 4, 2, 1, bias=False),
                nn.BatchNorm2d(outgoing),
                nn.ReLU(True),
            ]
        layers += [
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for incoming, outgoing in [(64, 128), (128, 256), (256, 512)]:
            layers += [
                nn.Conv2d(incoming, outgoing, 4, 2, 1, bias=False),
                nn.BatchNorm2d(outgoing),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [nn.Conv2d(512, 1, 4, 1, 0, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, images):
        return self.net(images).view(-1)

def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)

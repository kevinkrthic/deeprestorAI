import torch
import torch.nn as nn


class ChannelAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(channels//8, channels, 1),
            nn.Sigmoid()
        )


    def forward(self,x):

        w = self.fc(
            self.pool(x)
        )

        return x*w


class ResidualAttentionBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            ),

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            )

        )

        self.attention = ChannelAttention(
            channels
        )


    def forward(self,x):

        out = self.conv(x)

        out = self.attention(out)

        return out+x


class DownBlock(nn.Module):

    def __init__(self,in_c,out_c):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_c,
                out_c,
                3,
                padding=1
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            ),

            ResidualAttentionBlock(out_c),

            ResidualAttentionBlock(out_c)

        )

        self.down = nn.Conv2d(
            out_c,
            out_c,
            2,
            stride=2
        )


    def forward(self,x):

        feature=self.block(x)

        return feature,self.down(feature)



class UpBlock(nn.Module):

    def __init__(self,in_c,skip_c,out_c):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_c,
            out_c,
            2,
            stride=2
        )


        self.block = nn.Sequential(

            nn.Conv2d(
                out_c+skip_c,
                out_c,
                3,
                padding=1
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            ),

            ResidualAttentionBlock(out_c)

        )


    def forward(self,x,skip):

        x=self.up(x)

        x=torch.cat(
            [x,skip],
            dim=1
        )

        return self.block(x)


class DeepRestorAI(nn.Module):

    def __init__(self):

        super().__init__()


        self.enc1 = DownBlock(
            1,
            32
        )


        self.enc2 = DownBlock(
            32,
            64
        )


        self.bottleneck = nn.Sequential(

            ResidualAttentionBlock(64),
            ResidualAttentionBlock(64)

        )


        self.dec2 = UpBlock(
            64,
            64,
            64
        )


        self.dec1 = UpBlock(
            64,
            32,
            32
        )


        self.super_res = nn.Sequential(

            nn.Conv2d(
                32,
                128,
                3,
                padding=1
            ),

            nn.PixelShuffle(2),

            nn.LeakyReLU(
                0.1,
                inplace=True
            )

        )


        self.output = nn.Conv2d(
            32,
            1,
            3,
            padding=1
        )


    def forward(self,x):

        s1,x=self.enc1(x)

        s2,x=self.enc2(x)

        x=self.bottleneck(x)

        x=self.dec2(x,s2)

        x=self.dec1(x,s1)

        x=self.super_res(x)

        x=self.output(x)

        return x


DeepRestoreAI = DeepRestorAI
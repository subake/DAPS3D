# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch.nn as nn
import torch.nn.functional as F

from tasks.semantic.modules.convlstm import ConvLSTM

class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return x + y


class resBlock_with_add(nn.Module):
    def __init__(self, conv, act, bn):
        super(resBlock_with_add, self).__init__()

        self.conv = conv
        self.act = act
        self.bn = bn

    def forward(self, x, y):
        res = self.conv(x)
        res = self.act(res)
        res = self.bn(res)
        return res + y


class Trans(nn.Module):
    def __init__(self, trans, trans_act, trans_bn):
        super(Trans, self).__init__()
        self.trans = trans
        self.trans_act = trans_act
        self.trans_bn = trans_bn

    def forward(self, x):
        upA = self.trans(x)
        upA = self.trans_act(upA)
        upA = self.trans_bn(upA)
        return upA


class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super(AttentionBlock, self).__init__()
        self.Wg = nn.Sequential(nn.Conv2d(f_g, f_int, kernel_size=1, padding=0, stride=1),
                                nn.BatchNorm2d(f_int))

        self.Wx = nn.Sequential(nn.Conv2d(f_l, f_int, kernel_size=1, padding=0, stride=1),
                                nn.BatchNorm2d(f_int))

        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, kernel_size=1, padding=0, stride=1),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())
        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3), stride=1):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size, padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size, padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA = self.bn1(resA)

        resA = self.conv3(resA)
        resA = self.act3(resA)
        resA = self.bn2(resA)
        return resA + shortcut


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=kernel_size, padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA = self.bn1(resA)

        resA = self.conv3(resA)
        resA = self.act3(resA)
        resA = self.bn2(resA)
        resA = shortcut + resA

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB

class RecurrentBlock(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """
    def __init__(self, out_ch, t=2, num_layers=1):
        super(RecurrentBlock, self).__init__()

        self.t = t

        self.rec_layer = ConvLSTM(out_ch, [out_ch for _ in range(num_layers)], (3, 3), num_layers, True, True, False)

    def forward(self, x):
        prev_shape = list(x.shape)

        s = [prev_shape[0] // self.t] + prev_shape
        s[1] = self.t

        x = x.reshape(s)
        x, _ = self.rec_layer(x)
        
        return x[0].reshape(prev_shape)

class RecResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True, t=2):
        super(RecResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)
        
        self.rec_block = RecurrentBlock(out_filters, t=t)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA = self.bn1(resA)

        resA = self.rec_block(resA)
        resA = shortcut + resA

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3),drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.trans = nn.ConvTranspose2d(in_filters, out_filters, kernel_size, stride=(2, 2), padding=1)
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm2d(out_filters)

        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(out_filters, out_filters, kernel_size, padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size, padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size, padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = self.trans(x)
        if upA.shape != skip.shape:
            upA = F.pad(upA, (0, 1, 0, 1), mode='replicate')
        upA = self.trans_act(upA)
        upA = self.trans_bn(upA)
        if self.drop_out:
            upA = self.dropout1(upA)
        upB = upA + skip
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE = self.bn1(upE)

        upE = self.conv2(upE)
        upE = self.act2(upE)
        upE = self.bn2(upE)

        upE = self.conv3(upE)
        upE = self.act3(upE)
        upE = self.bn3(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE

class RecUpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3),drop_out=True, t=2):
        super(RecUpBlock, self).__init__()
        self.drop_out = drop_out
        self.trans = nn.ConvTranspose2d(in_filters, out_filters, kernel_size, stride=(2, 2), padding=1)
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm2d(out_filters)

        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(out_filters, out_filters, kernel_size, padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.rec_block1 = RecurrentBlock(out_filters, t=t)
        self.rec_block2 = RecurrentBlock(out_filters, t=t)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = self.trans(x)
        if upA.shape != skip.shape:
            upA = F.pad(upA, (0, 1, 0, 1), mode='replicate')
        upA = self.trans_act(upA)
        upA = self.trans_bn(upA)
        if self.drop_out:
            upA = self.dropout1(upA)
        upB = upA + skip
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE = self.bn1(upE)

        upE = self.rec_block1(upE)

        upE = self.rec_block2(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE


class SalsaNetRecLSTM(nn.Module):
    def __init__(self, nclasses):
        super(SalsaNetRecLSTM, self).__init__()
        self.nclasses = nclasses

        # WITH remission
        # self.downCntx = ResContextBlock(5, 32)
        
        # WITHOUT remission
        self.downCntx = ResContextBlock(4, 32)
        
        self.resBlock1 = RecResBlock(32, 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = RecResBlock(32, 2 * 32, 0.2, pooling=True)
        self.resBlock3 = RecResBlock(2 * 32, 4 * 32, 0.2, pooling=True)
        self.resBlock4 = RecResBlock(4 * 32, 8 * 32, 0.2, pooling=True)
        self.resBlock5 = RecResBlock(8 * 32, 16 * 32, 0.2, pooling=True)
        self.resBlock6 = RecResBlock(16 * 32, 16 * 32, 0.2, pooling=False)

        self.upBlock1 = RecUpBlock(16 * 32, 16 * 32, 0.2)
        self.upBlock2 = RecUpBlock(16 * 32, 8 * 32, 0.2)
        self.upBlock3 = RecUpBlock(8 * 32, 4 * 32, 0.2)
        self.upBlock4 = RecUpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock5 = RecUpBlock(2 * 32, 32, 0.2, drop_out=False)

        self.logits = nn.Conv2d(32, nclasses, kernel_size=(1, 1))

    def forward(self, x):
        downCntx = self.downCntx(x)
        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)
        down5b = self.resBlock6(down4c)

        up4e = self.upBlock1(down5b, down4b)
        up3e = self.upBlock2(up4e, down3b)
        up2e = self.upBlock3(up3e, down2b)
        up1e = self.upBlock4(up2e, down1b)
        up0e = self.upBlock5(up1e, down0b)

        logits = self.logits(up0e)
        logits = F.softmax(logits, dim=1)
        return logits

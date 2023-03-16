#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 12:05:39 2023

@author: ben
"""

import numpy as np
import os.path, random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd 
import torch.utils.data 
from torch.utils.data import Dataset
import pylab
import matplotlib.pyplot as plt

# get training data
dir = "./"
# download
if not os.path.isfile('data-airfoils.npz'): 
    import requests
    print("Downloading training data (300MB), this can take a few minutes the first time...") 
    with open("data-airfoils.npz", 'wb') as datafile:
        resp = requests.get('https://dataserv.ub.tum.de/s/m1615239/download?path=%2F&files=dfp-data-400.npz', verify=False)
        datafile.write(resp.content)

    
    
npfile=np.load(dir+'data-airfoils.npz')
print("Loaded data, {} training, {} validation samples".format(len(npfile["inputs"]),len(npfile["vinputs"])))
print("Size of the inputs array: "+format(npfile["inputs"].shape))


# helper to show three target channels: normalized, with colormap, side by side
def showSbs(a1,a2, stats=False, bottom="NN Output", top="Reference", title=None):
    c=[]
    for i in range(3):
        b = np.flipud( np.concatenate((a2[i],a1[i]),axis=1).transpose()) 
        min, mean, max = np.min(b), np.mean(b), np.max(b);
        if stats: print("Stats %d: "%i + format([min,mean,max]))
        b -= min; b /= (max-min)
        c.append(b)
    cm = 1/2.54
    fig, axes = pylab.subplots(1, 1, figsize=(12*cm, 6.5*cm))
    axes.set_xticks([]); axes.set_yticks([]);
    im = axes.imshow(np.concatenate(c,axis=1), origin='upper', cmap='magma')
    pylab.colorbar(im); pylab.xlabel('p, ux, uy'); 
    pylab.ylabel('%s           %s'%(bottom,top))
    plt.savefig('image.pdf',bbox_inches='tight')
NUM=72
showSbs(npfile["inputs"][NUM],npfile["targets"][NUM], stats=False, bottom="Labels", top="Eingangsdaten", title=None)
    
    
    
# number of training epochs
EPOCHS = 100
# batch size 
BATCH_SIZE = 10 
# learning rate 
LR = 0.00002
class DfpDataset():
    def __init__(self, inputs,targets):
        self.inputs  = inputs
        self.targets = targets
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
tdata = DfpDataset(npfile["inputs"],npfile["targets"])
vdata = DfpDataset(npfile["vinputs"],npfile["vtargets"])
trainLoader = torch.utils.data.DataLoader(tdata, batch_size=BATCH_SIZE, shuffle=True , drop_last=True)
valiLoader  = torch.utils.data.DataLoader(vdata, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

print("Training & validation batches: {} , {}".format(len(trainLoader),len(valiLoader) ))


    
def blockUNet(in_c, out_c, name, size=4, pad=1, transposed=False, bn=True, activation=True, relu=True, dropout=0. ):
    """
    A building block for a U-Net neural network 
    architecture. This class creates a 
    convolutional block that takes in an input 
    tensor, applies a convolutional layer,
    performs batch normalization, and applies 
    an activation function. Optionally,
    dropout can also be added to the block. 
    If transposed convolutional layers are
    desired, the user can specify this in the 
    function arguments.

    Parameters
    ----------
    in_c : int 
        Number of input channels.
    out_c : int
        Number of output channels.
    name : str
        Name of the block.
    size : int
        Kernel size for convolutional layers. 
    pad : int
        Padding size for convolutional layers.
    transposed : bool
        Whether to use transposed convolutional 
        layers.
    bn : bool
        Whether to apply batch normalization.
    activation : bool
        Whether to apply an activation function.
    relu : bool 
        Whether to use ReLU activation function. 
        If False, LeakyReLU is used. 
    dropout : float
        Dropout probability. If 0, dropout is 
        not applied.

    Returns:
    -------
        block : nn.Sequential
            A PyTorch Sequential object 
            representing the block.
    """
    block = nn.Sequential()

    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2, mode='bilinear'))
        # reduce kernel size by one for the upsampling (ie decoder part)
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))

    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout>0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d( dropout, inplace=True))

    if activation:
        if relu:
            block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
        else:
            block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    return block

class DfpNet(nn.Module):
    """
    A PyTorch module implementing the DfpNet 
    architecture.

    Parameters:
    ----------
    channelExponent : int
        The exponent used to calculate the number 
        of channels in each layer of the encoder.
    dropout : float
        The probability of dropout. 

    Methods:
    -------
    forward(x)
        Computes the forward pass of the DfpNet 
        module.
    """
    def __init__(self, channelExponent=6, dropout=0.):
        super(DfpNet, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = blockUNet(3         , channels*1, 'enc_layer1', transposed=False, bn=True, relu=False, dropout=dropout )
        self.layer2 = blockUNet(channels  , channels*2, 'enc_layer2', transposed=False, bn=True, relu=False, dropout=dropout )
        self.layer3 = blockUNet(channels*2, channels*2, 'enc_layer3', transposed=False, bn=True, relu=False, dropout=dropout )
        self.layer4 = blockUNet(channels*2, channels*4, 'enc_layer4', transposed=False, bn=True, relu=False, dropout=dropout )
        self.layer5 = blockUNet(channels*4, channels*8, 'enc_layer5', transposed=False, bn=True, relu=False, dropout=dropout ) 
        self.layer6 = blockUNet(channels*8, channels*8, 'enc_layer6', transposed=False, bn=True, relu=False, dropout=dropout , size=2,pad=0)
        self.layer7 = blockUNet(channels*8, channels*8, 'enc_layer7', transposed=False, bn=True, relu=False, dropout=dropout , size=2,pad=0)
     
        # note, kernel size is internally reduced by one for the decoder part
        self.dlayer7 = blockUNet(channels*8, channels*8, 'dec_layer7', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer6 = blockUNet(channels*16,channels*8, 'dec_layer6', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer5 = blockUNet(channels*16,channels*4, 'dec_layer5', transposed=True, bn=True, relu=True, dropout=dropout ) 
        self.dlayer4 = blockUNet(channels*8, channels*2, 'dec_layer4', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer3 = blockUNet(channels*4, channels*2, 'dec_layer3', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dec_layer2', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer1 = blockUNet(channels*2, 3         , 'dec_layer1', transposed=True, bn=False, activation=False, dropout=dropout )

    def forward(self, x):
        # note, this Unet stack could be allocated with a loop, of course... 
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        # ... bottleneck ...
        dout6 = self.dlayer7(out7)
        dout6_out6 = torch.cat([dout6, out6], 1)
        dout6 = self.dlayer6(dout6_out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# channel exponent to control network size
EXPO = 3

# setup network
net = DfpNet(channelExponent=EXPO)
#print(net) # to double check the details...

nn_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in nn_parameters])

# crucial parameter to keep in view: how many parameters do we have?
print("Trainable params: {}   -> crucial! always keep in view... ".format(params)) 

net.apply(weights_init)

criterionL1 = nn.L1Loss()
optimizerG = optim.Adam(net.parameters(), lr=LR, betas=(0.5, 0.999), weight_decay=0.0)

targets = torch.autograd.Variable(torch.FloatTensor(BATCH_SIZE, 3, 128, 128))
inputs  = torch.autograd.Variable(torch.FloatTensor(BATCH_SIZE, 3, 128, 128))






history_L1 = []
history_L1val = []

if os.path.isfile("network"):
  print("Found existing network, loading & skipping training")
  net.load_state_dict(torch.load("network")) # optionally, load existing network

else:
  print("Training from scratch")
  for epoch in range(EPOCHS):
      net.train()
      L1_accum = 0.0
      for i, traindata in enumerate(trainLoader, 0):
          inputs_curr, targets_curr = traindata
          inputs.data.copy_(inputs_curr.float())
          targets.data.copy_(targets_curr.float())

          net.zero_grad()
          gen_out = net(inputs)

          lossL1 = criterionL1(gen_out, targets)
          lossL1.backward()
          optimizerG.step()
          L1_accum += lossL1.item()

      # validation
      net.eval()
      L1val_accum = 0.0
      for i, validata in enumerate(valiLoader, 0):
          inputs_curr, targets_curr = validata
          inputs.data.copy_(inputs_curr.float())
          targets.data.copy_(targets_curr.float())

          outputs = net(inputs)
          outputs_curr = outputs.data.cpu().numpy()

          lossL1val = criterionL1(outputs, targets)
          L1val_accum += lossL1val.item()

      # data for graph plotting
      history_L1.append( L1_accum / len(trainLoader) )
      history_L1val.append( L1val_accum / len(valiLoader) )

      if epoch<3 or epoch%20==0:
          print( "Epoch: {}, L1 train: {:7.5f}, L1 vali: {:7.5f}".format(epoch, history_L1[-1], history_L1val[-1]) )

  torch.save(net.state_dict(), "network" )
  print("Training done, saved network")
  
  
l1train = np.asarray(history_L1)
l1vali  = np.asarray(history_L1val)
cm = 1/2.54  # centimeters in inches
plt.figure(figsize=((9*cm, 5*cm)),dpi=600)
plt.rcParams["font.family"] = "Times New Roman"
plt.grid(True)
plt.plot(np.arange(l1train.shape[0]),l1train,'b',label='Training')
plt.plot(np.arange(l1vali.shape[0] ),l1vali ,'g',label='Validierung')
plt.xlabel("Epochen")
plt.ylabel("Fehler")
plt.legend()
plt.savefig('image.pdf',bbox_inches='tight')
plt.show()


if not os.path.isfile('data-airfoils-test.npz'):
  import urllib.request
  url="https://physicsbaseddeeplearning.org/data/data_test.npz"
  print("Downloading test data, this should be fast...")
  urllib.request.urlretrieve(url, 'data-airfoils-test.npz')

nptfile=np.load('data-airfoils-test.npz')
print("Loaded {}/{} test samples\n".format(len(nptfile["test_inputs"]),len(nptfile["test_targets"])))

testdata = DfpDataset(nptfile["test_inputs"],nptfile["test_targets"])
testLoader  = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False, drop_last=True) 

net.eval()
L1t_accum = 0.
for i, validata in enumerate(testLoader, 0):
    inputs_curr, targets_curr = validata
    inputs.data.copy_(inputs_curr.float())
    targets.data.copy_(targets_curr.float())

    outputs = net(inputs)
    outputs_curr = outputs.data.cpu().numpy()

    lossL1t = criterionL1(outputs, targets)
    L1t_accum += lossL1t.item()
    if i<1: showSbs(targets_curr[0] , outputs_curr[0], bottom="Vorhersagen", top="Labels" , title=None)

print("\nAverage test error: {}".format( L1t_accum/len(testLoader) ))


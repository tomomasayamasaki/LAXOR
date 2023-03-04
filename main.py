"""
MIT License

Copyright (c) 2022 Tomomasa Yamasaki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

####################################################
# Import package                                   #
####################################################
import numpy as np
import time

# original
from LAXOR_Sim.Areca import Areca
import LAXOR_Sim.Tool as tool

start = time.time()
####################################################
# Initilazation                                    #
####################################################
areca = Areca()

####################################################
# Load image for inference                         #
####################################################
# this image shape is (1,3,32,32)
# original image is cat_label3.jpeg
# original image is converted into Conv1_in as theano format
img = np.load('./Pre-trained_model/Conv1_in.npz')
img = img['arr_0']

####################################################
# Conv 1 layer                                     #
####################################################
# load weights
w_path = './Pre-trained_model/Conv1_binary-weights.npy'
bias_path = './Pre-trained_model/Conv1_bias.npy'
beta_path = './Pre-trained_model/Conv1_BNbeta.npy'
gamma_path = './Pre-trained_model/Conv1_BNgamma.npy'
mean_path = './Pre-trained_model/Conv1_BNmean.npy'
std_path = './Pre-trained_model/Conv1_BNstd.npy'
w1, b1, beta1, gamma1, mean1, std1 = tool.load_npys(w_path, bias_path, beta_path, gamma_path, mean_path, std_path)


# input
input = img

# re-flip
w1 = tool.filpWeight_lasagne(w1)

# inverse std of BN
std1 = tool.inverseStd_lasagne(std1)


# output Simulator
out = areca.CPU_Binary_Conv2D(
                              input,
                              w1,
                              b1,
                              1,
                              0,
                              'ON',
                              BN_gamma=gamma1,
                              BN_beta=beta1,
                              BN_mean=mean1,
                              BN_std=std1)


####################################################
# Conv 2 layer                                     #
####################################################
# load weights
w_path = './Pre-trained_model/Conv2_binary-weights.npy'
bias_path = './Pre-trained_model/Conv2_bias.npy'
beta_path = './Pre-trained_model/Conv2_BNbeta.npy'
gamma_path = './Pre-trained_model/Conv2_BNgamma.npy'
mean_path = './Pre-trained_model/Conv2_BNmean.npy'
std_path = './Pre-trained_model/Conv2_BNstd.npy'
w2, b2, beta2, gamma2, mean2, std2 = tool.load_npys(w_path, bias_path, beta_path, gamma_path, mean_path, std_path)

# input
input = out

# re-flip
w2 = tool.filpWeight_lasagne(w2)

# inverse std of BN
std2 = tool.inverseStd_lasagne(std2)


# output Simulator
out = areca.Binary_Conv2D(input, w2, b2, 1, 0, 'ON', gamma2, beta2, mean2, std2)


####################################################
# Conv 3 layer                                     #
####################################################
# load weights
w_path = './Pre-trained_model/Conv3_binary-weights.npy'
bias_path = './Pre-trained_model/Conv3_bias.npy'
beta_path = './Pre-trained_model/Conv3_BNbeta.npy'
gamma_path = './Pre-trained_model/Conv3_BNgamma.npy'
mean_path = './Pre-trained_model/Conv3_BNmean.npy'
std_path = './Pre-trained_model/Conv3_BNstd.npy'
w3, b3, beta3, gamma3, mean3, std3 = tool.load_npys(w_path, bias_path, beta_path, gamma_path, mean_path, std_path)

# input
input = out

# re-flip
w3 = tool.filpWeight_lasagne(w3)

# inverse std of BN
std3 = tool.inverseStd_lasagne(std3)

# output Simulator
out = areca.Binary_Conv2D(input, w3, b3, 1, 0, 'ON', gamma3, beta3, mean3, std3)


####################################################
# Conv 4 layer                                     #
####################################################
# load weights
w_path = './Pre-trained_model/Conv4_binary-weights.npy'
bias_path = './Pre-trained_model/Conv4_bias.npy'
beta_path = './Pre-trained_model/Conv4_BNbeta.npy'
gamma_path = './Pre-trained_model/Conv4_BNgamma.npy'
mean_path = './Pre-trained_model/Conv4_BNmean.npy'
std_path = './Pre-trained_model/Conv4_BNstd.npy'
w4, b4, beta4, gamma4, mean4, std4 = tool.load_npys(w_path, bias_path, beta_path, gamma_path, mean_path, std_path)

# input
input = out

# re-flip
w4 = tool.filpWeight_lasagne(w4)

# inverse std of BN
std4 = tool.inverseStd_lasagne(std4)

# output Simulator
out = areca.Binary_Conv2D(input, w4, b4, 1, 0, 'ON', gamma4, beta4, mean4, std4)


####################################################
# Max pooling 4 layer                              #
####################################################
input = out

out = areca.MaxPooling(input,2,2)



####################################################
# Conv 5 layer                                     #
####################################################
# load weights
w_path = './Pre-trained_model/Conv5_binary-weights.npy'
bias_path = './Pre-trained_model/Conv5_bias.npy'
beta_path = './Pre-trained_model/Conv5_BNbeta.npy'
gamma_path = './Pre-trained_model/Conv5_BNgamma.npy'
mean_path = './Pre-trained_model/Conv5_BNmean.npy'
std_path = './Pre-trained_model/Conv5_BNstd.npy'
w5, b5, beta5, gamma5, mean5, std5 = tool.load_npys(w_path, bias_path, beta_path, gamma_path, mean_path, std_path)

# input
input = out

# re-flip
w5 = tool.filpWeight_lasagne(w5)

# inverse std of BN
std5 = tool.inverseStd_lasagne(std5)

# output Simulator
out = areca.Binary_Conv2D(input, w5, b5, 1, 0, 'ON', gamma5, beta5, mean5, std5)


####################################################
# Conv 6 layer                                     #
####################################################
# load weights
w_path = './Pre-trained_model/Conv6_binary-weights.npy'
bias_path = './Pre-trained_model/Conv6_bias.npy'
beta_path = './Pre-trained_model/Conv6_BNbeta.npy'
gamma_path = './Pre-trained_model/Conv6_BNgamma.npy'
mean_path = './Pre-trained_model/Conv6_BNmean.npy'
std_path = './Pre-trained_model/Conv6_BNstd.npy'
w6, b6, beta6, gamma6, mean6, std6 = tool.load_npys(w_path, bias_path, beta_path, gamma_path, mean_path, std_path)

# input
input = out

# re-flip
w6 = tool.filpWeight_lasagne(w6)

# inverse std of BN
std6 = tool.inverseStd_lasagne(std6)

# output Simulator
out = areca.Binary_Conv2D(input, w6, b6, 1, 0, 'ON', gamma6, beta6, mean6, std6)



####################################################
# Max pooling 6 layer                              #
####################################################
input = out

out = areca.MaxPooling(input,2,2)



####################################################
# Conv 7 layer                                     #
####################################################
# load weights
w_path = './Pre-trained_model/Conv7_binary-weights.npy'
bias_path = './Pre-trained_model/Conv7_bias.npy'
beta_path = './Pre-trained_model/Conv7_BNbeta.npy'
gamma_path = './Pre-trained_model/Conv7_BNgamma.npy'
mean_path = './Pre-trained_model/Conv7_BNmean.npy'
std_path = './Pre-trained_model/Conv7_BNstd.npy'
w7, b7, beta7, gamma7, mean7, std7 = tool.load_npys(w_path, bias_path, beta_path, gamma_path, mean_path, std_path)

# input
input = out

# re-flip
w7 = tool.filpWeight_lasagne(w7)

# inverse std of BN
std7 = tool.inverseStd_lasagne(std7)

# output Simulator
out = areca.Binary_Conv2D(input, w7, b7, 1, 0, 'ON', gamma7, beta7, mean7, std7)



####################################################
# Conv 8 layer                                     #
####################################################
# load weights
w_path = './Pre-trained_model/Conv8_binary-weights.npy'
bias_path = './Pre-trained_model/Conv8_bias.npy'
beta_path = './Pre-trained_model/Conv8_BNbeta.npy'
gamma_path = './Pre-trained_model/Conv8_BNgamma.npy'
mean_path = './Pre-trained_model/Conv8_BNmean.npy'
std_path = './Pre-trained_model/Conv8_BNstd.npy'
w8, b8, beta8, gamma8, mean8, std8 = tool.load_npys(w_path, bias_path, beta_path, gamma_path, mean_path, std_path)

# input
input = out

# re-flip
w8 = tool.filpWeight_lasagne(w8)

# inverse std of BN
std8 = tool.inverseStd_lasagne(std8)

# output Simulator
out = areca.Binary_Conv2D(input, w8, b8, 1, 0,'ON', gamma8, beta8, mean8, std8)




####################################################
# FC layer                                         #
####################################################
# load weights
w_path = './Pre-trained_model/FC9_binary-weights.npy'
bias_path = './Pre-trained_model/FC9_bias.npy'
w9, b9= tool.load_npys(w_path, bias_path)

# input
input = out
out = areca.Binary_FullyConnected(input, w9, b9)


end = time.time()
print('Run time: {} [s]'.format(np.round(end-start, 1)))

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


from LAXOR_Sim.Accelerator import Accelerator_Engine as AccEng
import numpy as np
import LAXOR_Sim.Tool as tool
import LAXOR_Sim.Config as config
import os

class Biscotto():


    def __init__(self):
        self.BitChips = AccEng()
        self.epsilon = config.EPSILON
        self.BufferSize_W = config.BUFFERSIZE_WEIGHTS
        self.BufferSize_I = config.BUFFERSIZE_INPUT
        self.BufferSize_B = config.BUFFERSIZE_BIAS
        self.PE_Num = config.PENUMS
        self.bitsize_pe = config.BIT_SIZE_PE
        self.OR_numbers = config.ORNUMS
        self.OR_bitwidth = config.ORBITWIDTH
        self.PINS_IW = config.PINS_IW
        self.NUM_LABELS = config.NUM_LABELS
        self.layer_count = 1
        self.mapping = 0  # initialize mapping storategy
        self.logfile_name = config.LOG_FILE
        if not os.path.exists('Sim Result'):
            os.makedirs('Sim Result')
        self.logfile_name = './Sim Result/'+self.logfile_name


        # error judgement
        if self.BufferSize_W < 0:
            q = 'BUFFERSIZE_WEIGHTS'
            print('[ERROR Config] {} should be positive value'.format(q))
            exit()

        if self.BufferSize_I < 0:
            q = 'BUFFERSIZE_INPUT'
            print('[ERROR Config] {} should be positive value'.format(q))
            exit()

        if self.BufferSize_B < 0:
            q = 'BUFFERSIZE_BIAS'
            print('[ERROR Config] {} should be positive value'.format(q))
            exit()

        if self.PE_Num < 0:
            q = 'PENUMS'
            print('[ERROR Config] {} should be positive value'.format(q))
            exit()

        if self.bitsize_pe < 0:
            q = 'BIT_SIZE_PE'
            print('[ERROR Config] {} should be positive value'.format(q))
            exit()

        if self.OR_numbers < 0:
            q = 'ORNUMS'
            print('[ERROR Config] {} should be positive value'.format(q))
            exit()

        if self.OR_bitwidth < 0:
            q = 'ORBITWIDTH'
            print('[ERROR Config] {} should be positive value'.format(q))
            exit()

        if self.PINS_IW < 0:
            q = 'PINS_IW'
            print('[ERROR Config] {} should be positive value'.format(q))
            exit()

        if self.NUM_LABELS < 0:
            q = 'NUM_LABELS'
            print('[ERROR Config] {} should be positive value'.format(q))
            exit()



    def _BinarizeHard(self, array):
        """
            Binarize the values
        """
        out = np.where(array>=0, 1, 0)
        return out



    def _Set2DformatWeights(self, num_filter):
        """
            Binarize the values
            Convert 4D matrix to 2D matrix.
            For example
                4D matrix (the number of filter, filter size, kernel size, kernel size)
                2D matrix (the number of filter, filter size * kernle size * kernle size)

                (256, 256, 3, 3) -> (256, 256*3*3)
        """
        L = num_filter
        self.weights = self.weights.reshape(L,self.unit_Row)
        M = np.where(self.weights>=0,1,0)



        if self.mapping == 'Part_PE':
            assert self.bitsize_pe > self.unit_Row
            RE = self.bitsize_pe-self.unit_Row
            A = int(np.ceil(RE/2))
            B = int(RE - A)
            ones = np.ones((L,A))
            zeros = np.zeros((L,B))
            M = np.concatenate([M, ones, zeros], 1)
            _,j = M.shape
            assert self.bitsize_pe == j

        elif self.mapping == 'Multi_PE':
            assert self.bitsize_pe < self.unit_Row
            W = M
            M = []
            if self.unit_Row%self.bitsize_pe != 0:
                RE = self.bitsize_pe - self.unit_Row%self.bitsize_pe
                A = int(np.ceil(RE/2))
                B = int(RE - A)
                ones = np.ones((L,A))
                zeros = np.zeros((L,B))
                W = np.concatenate([W, ones, zeros], 1)
            for l in range(L):
                for n in range(self.Npe):
                    y1 = self.bitsize_pe*n
                    y2 = y1 + self.bitsize_pe
                    M.append(W[l,y1:y2])
            M = np.array(M)

        self.weights = M



    def _Set2DformatInput(self, C, H, K):
        """   　
            Convert input format for hardware

            Convert 4D to 2D being (N, filter size * kernle size * kernle size)
            For example

        """
        if K > 0: # Convolution
            mat = self.input.reshape(C,H,H)
            L = int(self.l*self.l)
            mat2D = np.zeros((L,self.unit_Row), dtype=int)

            # convert format for hardware
            l_idx = 0
            for h in range(self.l):
                hv = h * self.stride
                for w in range(self.l):
                    arr = []
                    wv = w * self.stride
                    for c in range(C):
                        for i in range(K):
                            for j in range(K):
                                l = mat[c,hv+i,wv+j]
                                arr.append(l)
                    mat2D[l_idx,:] = arr
                    l_idx += 1
            M = mat2D

        elif K == 0: #fully connected
            if H > 0:
                L = 1 # batch size = 1
                self.unit_Row = C * H * H
                mat2D = self.input.reshape(L,self.unit_Row)
                M = mat2D
            else:
                L = 1
                mat2D = self.input
                M = mat2D

        if self.mapping == 'Part_PE':
            assert self.bitsize_pe > self.unit_Row
            RE = self.bitsize_pe-self.unit_Row
            zeros = np.zeros((L,RE))
            M = np.concatenate([mat2D, zeros], 1)


        elif self.mapping == 'Multi_PE':
            assert self.bitsize_pe < self.unit_Row
            I = mat2D
            M = []
            if self.unit_Row%self.bitsize_pe != 0:
                RE = self.bitsize_pe - self.unit_Row%self.bitsize_pe
                zeros = np.zeros((L,RE))
                I = np.concatenate([I, zeros], 1)
            for l in range(L):
                for n in range(self.Npe):
                    y1 = self.bitsize_pe*n
                    y2 = y1 + self.bitsize_pe
                    M.append(I[l,y1:y2])
            M = np.array(M)

        self.input = M



    def _QuatizeBias(self):
        """
            Convert bias into 9-bits
        """
        self.bias = np.floor(self.bias)


    ##################################################################################################
    # API
    #    Binary Convolution Layer
    #    name: Binary_Conv2D
    #    parameters:
    #        1. input
    #        2. weights
    #        3. bias
    #        4. stride
    #        5. padding
    #        6. BatchNorm
    #        7. BN_gamma
    #        8. BN_beta
    #        9. BN_mean
    #       10. BN_std
    ##################################################################################################
    def Binary_Conv2D(self, input, weights, bias, stride=1, padding=0, BatchNorm='OFF', BN_gamma=None, BN_beta=None, BN_mean=None, BN_std=None):

        f = open(self.logfile_name, 'a')
        txt = '\n██ No.{0:2} Layer ██ {1:20} ██\n'.format(self.layer_count, 'Binary Convolution')
        print(txt)
        f.write('\n')
        f.write(txt)
        f.close()
        self.layer_count += 1


        # set
        self.input = input
        self.weights = weights
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.BN_gamma = BN_gamma
        self.BN_beta = BN_beta
        self.BN_mean = BN_mean
        self.BN_std = BN_std


        # values
        B, C, H, _ = self.input.shape
        F, S, K, _ = self.weights.shape
        self.unit_Row = S * K * K
        self.l = int((H-K)/self.stride + 1) # the size of output
        datasize_output = F * self.l * self.l


        # select mapping storategy
        map = self.unit_Row - self.bitsize_pe
        self.mapping = np.where(map<=0, np.where(map==0, 'Single_PE', 'Part_PE'), 'Multi_PE')
        self.Npe = int(np.ceil(self.unit_Row/self.bitsize_pe)) # The number of PE which one convolution needs

        # Padding
        if self.padding > 0:
            pad = np.zeros((B, C, H+2*self.padding, H+2*self.padding))
            pad[:,:,self.padding:-self.padding,self.padding:-self.padding] = self.input
            self.input = pad
            B, C, H, _ = self.input.shape
            self.l = int((H-K)/self.stride + 1) # the size of output
            datasize_output = F * self.l * self.l


        # reshape weights and input, and convert weights to 0 or 1
        self._Set2DformatWeights(F) # weights (numpy.ndarray)
        self._Set2DformatInput(C, H, K) # input (numpy.ndarray)
        self._QuatizeBias() # quantation for bias


        # Compute convolution
        out = self.BitChips.Binary_Conv2D(self.input, self.weights, self.bias, datasize_output, self.mapping, self.unit_Row, self.l)

        if BatchNorm == 'ON':
            out = self.BatchNorm(out,self.BN_gamma,self.BN_beta,self.BN_mean,self.BN_std)

        if datasize_output != config.NUM_LABELS:
            out = tool.binarize_hardbase(out)

        return out



    def Binary_FullyConnected(self, input, weights, bias, BatchNorm='OFF', BN_gamma=None, BN_beta=None, BN_mean=None, BN_std=None):
        """
            Computes binarized fully connected
            Computed by BNN Accelerator

            *parameters*
            1. input:       Binary input [0, 1] for binarized convolution layer        shape:(1,256,4,4)
            2. weights:     binary weights [-1, 1] for binarized convolution layer.    shape: (4096, 10)
            3. bias:        bias for binarized convolution layer.       shape: (10)
        """
        f = open(self.logfile_name, 'a')
        txt = '\n██ No.{0:2} Layer ██ {1:20} ██\n'.format(self.layer_count, 'Binary Fully Connected')
        print(txt)
        f.write('\n')
        f.write(txt)
        f.close()
        self.layer_count += 1


        # set
        self.input = input
        self.weights = weights
        self.weights = np.transpose(self.weights) # (ch, the number of filter) → (the number of filter, ch)
        self.bias = bias
        self.BN_gamma = BN_gamma
        self.BN_beta = BN_beta
        self.BN_mean = BN_mean
        self.BN_std = BN_std

        outputsize, self.unit_Row = self.weights.shape

        # values
        if self.input.ndim == 4:
            _,C,H,_ = self.input.shape

            # select mapping storategy
            map =  self.unit_Row - self.bitsize_pe
            self.mapping = np.where(map<=0, np.where(map==0, 'Single_PE', 'Part_PE'), 'Multi_PE')
            self.Npe = int(np.ceil(self.unit_Row/self.bitsize_pe))

            # convert data format
            self._Set2DformatWeights(outputsize) # Weights
            self._Set2DformatInput(C, H, 0) #input
            self._QuatizeBias() #bias

        else:
            # select mapping storategy
            map =  self.unit_Row - self.bitsize_pe
            self.mapping = np.where(map<=0, np.where(map==0, 'Single_PE', 'Part_PE'), 'Multi_PE')
            self.Npe = int(np.ceil(self.unit_Row/self.bitsize_pe))

            _,C = self.input.shape
            self._Set2DformatWeights(outputsize) # Weights
            self._Set2DformatInput(C, 0, 0) #input
            self._QuatizeBias() #bias

        # Compute
        out = self.BitChips.Binary_FullyConnected(self.input, self.weights, self.bias, self.mapping, self.unit_Row)

        if BatchNorm == 'ON':
            out = self.BatchNorm(out,self.BN_gamma,self.BN_beta,self.BN_mean,self.BN_std)

        if outputsize != config.NUM_LABELS:
            out = tool.binarize_hardbase(out)

        return out



    def MaxPooling(self, input, ksize, stride):
        """
            Max Pooling layer
            Computed by BNN Accelerator
        """
        f = open(self.logfile_name, 'a')
        txt = '\n██ No.{0:2} Layer ██ {1:20} ██\n'.format(self.layer_count, 'Max Pooling')
        print(txt)
        f.write('\n')
        f.write(txt)
        f.close()
        self.layer_count += 1

        # set
        self.input = input
        self.ksize = ksize
        self.stride = stride
        _, _, H, _ = self.input.shape

        # error judgement
        """
        if (H-self.ksize)%self.stride != 0:
            print('[ERROR No.{} Max Pooling]\nKernel size:{}\nStride:{}\nCheck your kernel size and stride values for No.{} max pooling. Your setup does not work.'.format(self.layer_count-1, self.ksize, self.stride, self.layer_count))
            exit()
        """

        if ksize*ksize > self.OR_bitwidth:
            print('[ERROR No.{} Max Pooling]\nKernel size:{}\nOR bit-width:{}\nBit width should be more than kernel size squared ({})'.format(self.layer_count-1, self.ksize, self.OR_bitwidth, self.ksize*self.ksize))
            exit()


        self.l = int((H-self.ksize)/self.stride + 1) # the size of output


        # compute
        out = self.BitChips.MaxPooling(self.input, self.ksize, self.stride, self.l)
        return out



    def CPU_Binary_Conv2D(self, input, weights, bias, stride=1, padding=0, BatchNorm='OFF', BN_gamma=None, BN_beta=None, BN_mean=None, BN_std=None):
        """
            Convolution for multi-channels with padding=0, strides=1

            *parameter*
            input (numpy): input array with shape (channel, height, width)
            weights (numpy): filter array with shape (fd, fd, in_depth, out_depth) with odd fd.
            bias (numpy)
        """
        f = open(self.logfile_name, 'a')
        txt = '\n██ No.{0:2} Layer ██ {1:20}\n'.format(self.layer_count, 'Binary Convolution (CPU)')
        print(txt)
        f.write('\n')
        f.write(txt)
        f.close()
        self.layer_count += 1
        print('This layer is computed by CPU.')

        # set
        self.input = input
        self.weights = weights
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.BN_gamma = BN_gamma
        self.BN_beta = BN_beta
        self.BN_mean = BN_mean
        self.BN_std = BN_std

        out = tool.Fconv2D(self.input, self.weights, self.bias, self.stride, self.padding)

        f = open(self.logfile_name, 'a')
        txt = '*** OUTPUT INFO ***\n'
        txt +=' Shape                     : {}\n'.format(out.shape)
        txt +='---------------------------------------------\n\n'
        print(txt)
        f.write(txt)
        f.close()

        if BatchNorm == 'ON':
            out = self.BatchNorm(out,self.BN_gamma,self.BN_beta,self.BN_mean,self.BN_std)

        out = self._BinarizeHard(out)

        return np.array(out)


    def CPU_pooling(self, input, type, ksize, stride):
        f = open(self.logfile_name, 'a')
        txt = '\n██ No.{0:2} Layer ██ {1:20} ██\n'.format(self.layer_count, type+' Pooling (CPU)')
        print(txt)
        f.write('\n')
        f.write(txt)
        f.close()
        self.layer_count += 1
        print('This layer is computed by CPU.')

        self.input = input
        self.type = type
        self.ksize = ksize
        self.stride = stride

        if self.type == 'ave':
            out = tool.Favepooling(self.input, self.ksize, self.stride)
        elif self.type == 'max':
            out = tool.Fmaxpooling(self.input, self.ksize, self.stride)
        else:
            print('[ERROR] Please select ave or max for CPU-based pooling layer')
            exit()

        f = open(self.logfile_name, 'a')
        txt = '*** OUTPUT INFO ***\n'
        txt +=' Shape                     : {}\n'.format(out.shape)
        txt +='---------------------------------------------\n\n'
        print(txt)
        f.write(txt)
        f.close()

        return out


    def BatchNorm(self, input, BN_gamma, BN_beta, BN_mean, BN_std):
        """
            batch normalization
        """
        f = open(self.logfile_name, 'a')
        txt = '\n██ No.{0:2} Layer ██ {1:20} ██\n'.format(self.layer_count, 'Batch Normalization')
        print(txt)
        f.write('\n')
        f.write(txt)
        f.close()
        self.layer_count += 1

        if np.array(input).ndim == 4:
            _,c,h,w = np.array(input).shape
            out = np.zeros((1,c,h,w))
            for i in range(c):
                out[0,i,:,:] = (input[:,i,:,:]-BN_mean[i])*(BN_gamma[i]/BN_std[i])+BN_beta[i]

        else:
            _,c = np.array(input).shape
            out = np.zeros((1,c))
            for i in range(c):
                out[0,i] = (input[:,i]-BN_mean[i])*(BN_gamma[i]/BN_std[i])+BN_beta[i]

        f = open(self.logfile_name, 'a')
        txt = '*** OUTPUT INFO ***\n'
        txt +=' Shape                     : {}\n'.format(out.shape)
        txt +='---------------------------------------------\n\n'
        print(txt)
        f.write(txt)
        f.close()

        return out

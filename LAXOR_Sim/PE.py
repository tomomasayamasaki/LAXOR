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


import numpy as np
import LAXOR_Sim.Config as config


class PE:

    def __init__(self):
        self.XNORout = 0
        self.weights = 0
        self.input = 0
        self.bias = 0
        self.flag = 1
        self.bitsize_pe = config.BIT_SIZE_PE



    def SetXNORWeights(self, subweights):
        self.weights = np.array(subweights) #1024bits



    def SetXNORInput(self, subinput):
        self.input = np.array(subinput) #1024bits



    def SetPopArray(self, array, mode='XNOR_output'):
        if mode == 'XNOR_output':
            self.XNORout = array
        elif mode == 'bias':
            self.bias = array
        else:
            print('SetPopArray@PE_XNOR_POP mode error')



    def _transferfunc(self, out):
        NumOne = out
        NumAll = self.bitsize_pe
        NumZero = NumAll - NumOne
        out2 = NumOne - NumZero
        return out2



    def RunXNOR(self):
        """
            Compute XNOR.
            self.out is 1024 bits
        """
        out = np.where(self.weights==self.input, 1, 0)
        self.SetPopArray(out,'XNOR_output')


    def RunPopcount(self):
        """
            Compute Popcount
            input for this is 1024bits
        """
        out = np.count_nonzero(self.XNORout == 1)
        out_tf = self._transferfunc(out)
        out_tf = float(out_tf) + self.bias
        return out_tf

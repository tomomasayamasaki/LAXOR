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

# original
from LAXOR_Sim.Areca import Areca

####################################################
# Initilazation                                    #
####################################################
areca = Areca()


# input
input = np.random.randint(-1, 1, (1,3,32,32))

a = 2
# bias
b64 = np.random.rand(64)
b128 = np.random.rand(128)
b256 = np.random.rand(256)
b512 = np.random.rand(512)

# weights
w1 = np.random.randint(-1, 1, (64,3,2,2))
w2 = np.random.randint(-1, 1, (64,64,2,2))
w3 = np.random.randint(-1, 1, (128,64,2,2))
w4 = np.random.randint(-1, 1, (128,128,2,2))
w5 = np.random.randint(-1, 1, (256,128,2,2))
w6 = np.random.randint(-1, 1, (256,256,2,2))
w7 = np.random.randint(-1, 1, (512,256,2,2))
w8 = np.random.randint(-1, 1, (512,512,2,2))

# BNN
out = areca.CPU_Binary_Conv2D(input,w1,b64)
out = areca.Binary_Conv2D(out,w2,b64)
out = areca.Binary_Conv2D(out,w3,b128)
out = areca.Binary_Conv2D(out,w4,b128)
out = areca.MaxPooling(out,2,2)


out = areca.Binary_Conv2D(out,w5,b256)
out = areca.Binary_Conv2D(out,w6,b256)
out = areca.Binary_Conv2D(out,w6,b256)
out = areca.Binary_Conv2D(out,w7,b512)
out = areca.Binary_Conv2D(out,w8,b512)
out = areca.Binary_Conv2D(out,w8,b512)
out = areca.MaxPooling(out,2,2)

out = areca.Binary_Conv2D(out,w8,b512)
out = areca.Binary_Conv2D(out,w8,b512)
out = areca.Binary_Conv2D(out,w8,b512)

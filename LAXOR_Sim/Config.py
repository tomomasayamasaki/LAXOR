import numpy as np
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


# Log file
LOG_FILE = 'Output.txt'

# PEs
PENUMS = 256 # the number of PEs
BIT_SIZE_PE = 1024 #bits  Bit size of PE (XNOR-POPCOUNT)

# OR logic
ORNUMS = 256
ORBITWIDTH = 4

# BUFFER
a = 1024
BUFFERSIZE_INPUT = a #bits
BUFFERSIZE_WEIGHTS = a #bits
BUFFERSIZE_BIAS = 9 # bits

# I/O
PINS_IW = 8 # the number of pins for input or weights buffer

# BATCH NORMALIZATION
"""
    if you trained your model by theano and lasagne, please EPSILON = 0
"""
EPSILON = 0

# CIFAR10
NUM_LABELS = 10 # the number of labels

# Unit Dynamic Energy
## Computation
ENERGY_POPCOUNT = 0.00054075
ENERGY_XOR = 2.7124E-07
ENERGY_OR = 2.25811E-06
ENERGY_BNA = 0.000437255
ENERGY_COMPARISON = 5.9293E-05

## Data movement
ENERGY_DM_READ_BUFFER_IW = 2.21987E-06
ENERGY_DM_LOAD_PE = 4.08203E-07
ENERGY_DM_REAM_BUFFER_BIAS = 5.01043E-06
ENERGY_DM_LOAD_CONTROL = 5.9393E-05


# Unit Leakage Energy
# Computation
LEAK_POPCOUNT = 500
LEAK_XOR = 10.25390625
LEAK_OR = 24.63866016
LEAK_BNA = 791.7637969
LEAK_COMPARISON = 509.1364333

## Data movement
LEAK_DM_READ_BUFFER_IW = 23.25170117
LEAK_DM_REAM_BUFFER_BIAS = 14.56696875
LEAK_DM_LOAD_CONTROL = 31.4090625


# Clock Period
CLOCK_PERIOD = 0.000000005

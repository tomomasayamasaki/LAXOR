<a href="https://istd.sutd.edu.sg/people/phd-students/tomomasa-yamasaki">
    <img src="https://github.com/tomomasayamasaki/LAXOR/blob/main/README/logo.png" alt="Tomo logo" title="Tomo" align="right" height="110" />
</a>

# LAXOR: A BNN Accelerator with Latch-XOR Logic for Local Computing
  
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![](https://img.shields.io/github/downloads/tomomasayamasaki/LAXOR/total)
![](https://img.shields.io/github/repo-size/tomomasayamasaki/LAXOR)
![](https://img.shields.io/github/commit-activity/y/tomomasayamasaki/LAXOR)
![](https://img.shields.io/github/last-commit/tomomasayamasaki/LAXOR)
![](https://img.shields.io/github/languages/count/tomomasayamasaki/LAXOR)

![gif](https://github.com/tomomasayamasaki/LAXOR/blob/main/README/LAXOR.gif)

## 🟨 Contents
- [Introduction](https://github.com/tomomasayamasaki/LAXOR#-introduction)
    - [LAXOR Accelerator](https://github.com/tomomasayamasaki/LAXOR#%EF%B8%8F-laxor-accelerator)
    - [LAXOR Accelerator Simulator](https://github.com/tomomasayamasaki/LAXOR#%EF%B8%8F-laxor-accelerator-simulator)
- [Repogitory File Structure](https://github.com/tomomasayamasaki/LAXOR#-repository-file-structure)
    - [LAXOR_Sim](https://github.com/tomomasayamasaki/LAXOR#%EF%B8%8F-laxor_sim)
    - [Program and pre-trained model for test running](https://github.com/tomomasayamasaki/LAXOR#%EF%B8%8F-program-and-pre-trained-model-for-test-running)
- [Installation Requirements](https://github.com/tomomasayamasaki/LAXOR#-installation-requirements)
- [How to Run](https://github.com/tomomasayamasaki/LAXOR#-how-to-run)
    - [Areca platform](https://github.com/tomomasayamasaki/LAXOR#%EF%B8%8F-areca-platform)]
    - [Load pre-trained model](https://github.com/tomomasayamasaki/LAXOR#%EF%B8%8F-load-pre-trained-model)
    - [Config](https://github.com/tomomasayamasaki/LAXOR#%EF%B8%8F-config)
    - [Run Example](https://github.com/tomomasayamasaki/LAXOR#%EF%B8%8F-run-example)
- [Citing LAXOR accelerator and simulator](https://github.com/tomomasayamasaki/LAXOR#-citing-laxor-accelerator-and-simulator)
- [Licence](https://github.com/tomomasayamasaki/LAXOR#-licence)

## 🟨 Introduction
### ◼️ LAXOR Accelerator
#### What is LAXOR Accelerator
LAXOR is an Binary Neural Network accelerator proposed by a group of people from [Singapore Univerisity of Technology and Design](https://www.sutd.edu.sg/) (SUTD) and [Institute of Microelectronics](https://www.a-star.edu.sg/ime/) (IME), Agency for Science, Technology and Research (A*STAR). The essence of LAXOR lies in a novel local computing paradigm that fuses the weight storage (i.e., latch) and the compute unit (i.e., XOR gate) in a single logic to minimize data movement, achieving 4.2× lower energy consumption. Assisted with the optimized population count circuits, LAXOR accelerator obtains an energy efficiency of 2299 T OPS/W , 3.4× ∼ 37.6× higher compared to the advanced BNN accelerator architectures.   

|   | LAXOR Accelerator |
| ------------- | ------------- |
| CMOS Technology  | 28nm  |
| Desing Type  | Digital  |
| Result Type  | Synthesis  |
| VDD ($V$)  | 0.5-0.9  |
| Bit Width  | 1  |
| Frequency ($MHz$)  | 200M  |
| Core Area ($mm^2$)  | 2.73  |
| Performance ($TOPS$)  | 104.8  |
| Compute Density ($TOPS/mm^2$)  | 38.388  |
| MAC Energy Efficiency ($TOPS/W$)  | 2299 @0.5  |
| Bit Accurate  | Yes  |

#### Many-core Architecture
LAXOR accelerator has a many-core architecture with compact XOR arrays and energy-efficient popcount logic for local computing. The architecture consists of 4 Processing Engine (PE) clusters, a global controller and configuration unit, an accumulation unit to realize total sums for activation layers or Fully-Connected (FC) layers, and a comparison block to determine the inference result based on the maximum value.

<p align="center"><img width=60% src="https://github.com/tomomasayamasaki/LAXOR/blob/main/README/LAXOR_Core.png"></p>

#### Dataflow Mapping
LAXOR is able to support BNN topologies with small kernel dimensions, ranging from $1×1$ to $11×11$ by a flexible dataflow mapping scheme. Tha mapping strategy is to leverage the parallelism of 4 clusters. For a 3D odd-sized Kernel Wi (e.g., $(2N + 1)^2×C$, where $C$ is the number of channels and $N$ is a non-negative integer), we can expand the square term into $(4N^2 + 4N + 4)×C − 3×C$ while 3 groups of ‘0’ are padded, each with a size of $C$. By doing so, we transform it into a $(4N^2 + 4N + 4)×C$ shape before equally splitting it into four segments and mapping them onto the corresponding PEi in the clusters. 

Example: a 3D $3×3×C$ Kernel. (Detail is on our paper)
<p align="center"><img width=60% src="https://github.com/tomomasayamasaki/LAXOR/blob/main/README/LAXOR_Mapping1.png"></p>
<p align="center"><img width=60% src="https://github.com/tomomasayamasaki/LAXOR/blob/main/README/LAXOR_Mapping2.png"></p>

#### Compact 10-Transistor Latch-XOR Computing Cell
On LAXOR accelerator, the bitwise multiplication efficiently with an inverted-XOR gate is implemented. Besides, the accumulation can be realized by a population count logic (i.e., popcount logic), which counts the number of '0' of the XOR output. Also, LAXOR accelerator consists of a proposed tightly coupled 10T Latch-XOR cell in which  the computation is in-situ with the data storage for local computing. It comprises a transmission gate, a cross-coupled latch, a two-transistor (i.e., M1, M2) switching path, and an inverter.

<p align="center"><img width=60% src="https://github.com/tomomasayamasaki/LAXOR/blob/main/README/LAXOR_10TXOR.png"></p>

#### Popcount Unit PCL Design
To further reduce data movement, we combine an array of 1024 Latch-XOR cells together with a Parallel-counter-Carry-Look-ahead (PCL) unit and a computation and activation unit to form a PE. This tightly-coupled configuration allows for efficient counting of the ‘0’s from the output of the array and generating activations or partial sums for further operation. The figure of a PCL unit which is deploied on LAXOR accelerator as follow.

<p align="center"><img width=60% src="https://github.com/tomomasayamasaki/LAXOR/blob/main/README/LAXOR_PCL.png"></p>

### ◼️ LAXOR Accelerator Simulator
We design a python-based simulator for the proposed LAXOR accelerator. The purpose of the simulator is to
- (1) map and verify the functionality of a BNN model onto the proposed architecture
- (2) generate application-specific, cycle-accurate results (e.g., latency, energy, utilization, etc.) for design analysis.

The LAXOR simulator consists of a front-end tool, Areca, and a back-end tool,Bits-Island. Areca interfaces with the pre-trained model and user configurations before generating the data stream in a format tailored to the accelerator. Bits-Island replicates the LAXOR architecture, maps the data stream onto different PEs, and simulates the functionality layer by layer. Eventually the tool-chain reports the mapping results, layer output, and critical design metrics by harnessing embedded cycle count, latency and energy models. To ensure accurate energy estimation, latency and energy per atomic hardware operation such as single XOR gate, buffer read, weight loading, are provided to Areca using Cadence Spectre gate-level and post layout simulations.     

<p align="center"><img width=80% src="https://github.com/tomomasayamasaki/LAXOR/blob/main/README/fig1.png"></p>

## 🟨 Repository File Structure
### ◼️ LAXOR_Sim
#### Areca (Front-end)
- [Areca.py](https://github.com/tomomasayamasaki/LAXOR/blob/main/LAXOR_Sim/Areca.py)

#### Bits-Island (Back-end)
- [Accelerator.py](https://github.com/tomomasayamasaki/LAXOR/blob/main/LAXOR_Sim/Accelerator.py)
- [Cycle_count.py](https://github.com/tomomasayamasaki/LAXOR/blob/main/LAXOR_Sim/Config.py)
- [Energy.py](https://github.com/tomomasayamasaki/LAXOR/blob/main/LAXOR_Sim/Energy.py)
- [PE.py](https://github.com/tomomasayamasaki/LAXOR/blob/main/LAXOR_Sim/PE.py)
- [Performance.py](https://github.com/tomomasayamasaki/LAXOR/blob/main/LAXOR_Sim/Performance.py)

#### others
- [Config.py](https://github.com/tomomasayamasaki/LAXOR/blob/main/LAXOR_Sim/Config.py)
- [Tool.py](https://github.com/tomomasayamasaki/LAXOR/blob/main/LAXOR_Sim/Tool.py)
    
### ◼️ Program and pre-trained model for test running 

- Pre-trained_model

    Pre-trained model of Binary CNN for CIFAR-10, which the accuracy is 85.25%, the total enegy is 3.82 $uJ$, and the model size is 0.51 $MB$.
    
- [Example_simple.py](https://github.com/tomomasayamasaki/LAXOR/blob/main/Example_simple.py)

    A program to run the LAXOR accelerator simulator with random weights.

- [main.py](https://github.com/tomomasayamasaki/LAXOR/blob/main/main.py)

    A program to run the LAXOR accelerator simulator with the pre-trained binary CNN model for CIFAR-10.

## 🟨 Installation Requirements
- Python version3
- datetime
- matplotlib
- torch
- numpy

## 🟨 How to Run
### ◼️ Areca platform
#### Import Areca
```python
from LAXOR_Sim.Areca import Areca
```

#### Initilaize Areca
```python
areca = Areca()
```

#### Binarized Convolution layer not computed by LAXOR accelerator
```python
# If you want to run a batch normalization after convolution, select BatchNorm='ON' and add parameters for batch normalization by numpy array format.
# input: numpy array
# weights: numpy array
# bias: numpy array
# padding: int

out = areca.CPU_Binary_Conv2D(input, weights, bias, padding=0, BatchNorm='OFF', BN_gamma=None, BN_beta=None, BN_mean=None, BN_std=None)
```

#### Binarized Convolution layer computed by LAXOR accelerator
```python
# If you want to run a batch normalization after convolution, select BatchNorm='ON' and add parameters for batch normalization by numpy array format.
# input: numpy array
# weights: numpy array
# bias: numpy array
# stride: int
# padding: int

out = areca.Binary_Conv2D(input, weights, bias, stride=1, padding=0, BatchNorm='OFF', BN_gamma=None, BN_beta=None, BN_mean=None, BN_std=None)
```

#### Binarized Fullyconnected layer
```python
# If you want to run a batch normalization after convolution, select BatchNorm='ON' and add parameters for batch normalization by numpy array format.
# input: numpy array
# weights: numpy array
# bias: numpy array

out = areca.Binary_FullyConnected(input, weights, bias, BatchNorm='OFF', BN_gamma=None, BN_beta=None, BN_mean=None, BN_std=None)
```

#### Max pooling layer
```python
# input: numpy array
# ksize: int
# stride: int

out = areca.MaxPooling(input, ksize, stride)
```

### ◼️ Load pre-trained model
This is one of examples to show how to load. User should store weights, input, bias, and some parameters into numpy array. The pre-trained model we provide is stored npy file. In order to load them, we use numpy.load().
```python
import LAXOR_Sim.Tool as tool

w_path = './Pre-trained_model/Conv1_binary-weights.npy'
bias_path = './Pre-trained_model/Conv1_bias.npy'
beta_path = './Pre-trained_model/Conv1_BNbeta.npy'
gamma_path = './Pre-trained_model/Conv1_BNgamma.npy'
mean_path = './Pre-trained_model/Conv1_BNmean.npy'
std_path = './Pre-trained_model/Conv1_BNstd.npy'

w1, b1, beta1, gamma1, mean1, std1 = tool.load_npys(w_path, bias_path, beta_path, gamma_path, mean_path, std_path)
```

#### Weights for binarized convolution layer
```python
w_path = './Pre-trained_model/Conv1_binary-weights.npy'
weights = np.load(w_path)
print(weights)

"""
[[[[ 1  1]
   [ 1  1]]

  [[ 1 -1]
   [ 1 -1]]

  [[-1 -1]
   [-1 -1]]]

 ...

 [[[ 1  1]
   [ 1 -1]]

  [[ 1 -1]
   [-1 -1]]

  [[ 1 -1]
   [ 1  1]]]]
"""
```

#### Bias for binarized convolution layer
```python
bias_path = './Pre-trained_model/Conv1_bias.npy'
bias = np.load(bias_path)
print(bias)

"""
[-1.53262466e-01 -8.07617307e-01 -5.98477423e-01  8.31361294e-01
 -3.19540091e-02 -9.77912545e-02 -7.92104006e-01  1.63953304e-01
 -7.66488433e-01  7.03036129e-01 -1.27875507e-01 -2.24554762e-01
  4.48263705e-01 -1.31150529e-01 -1.73672631e-01 -1.33374967e-02
   ...
  1.89023882e-01 -2.55141824e-01 -8.33954439e-02  1.33623332e-02
 -6.80823684e-01  1.56198531e-01  2.09271386e-01  1.42073661e-01
 -1.16940970e-02  6.63392007e-01 -3.19188684e-01 -4.96945649e-01
  1.12402477e-05 -1.28726274e-01 -6.90906346e-01  4.38432664e-01]
"""
```

### ◼️ Config
#### The name of log file
```python
LOG_FILE = 'Output.txt'
```

#### Number of PEs (XOR & Popcount)
```python
PENUMS = 256
```

#### Bit size of PE (XOR & Popcount)
```python
BIT_SIZE_PE = 1024 #bit
```

#### OR Logic
```python
ORNUMS = 256 # number of OR-Logic
ORBITWIDTH = 4 # Bit width of OR-Logic
```

#### Buffer size
```python
# input buffer
BUFFERSIZE_INPUT = 1024 #bits
# weights buffer
BUFFERSIZE_WEIGHTS = 1024 #bits
# bias buffer
BUFFERSIZE_BIAS = 9 # bits
```

#### The number of pins for input or weights buffer
```python
PINS_IW = 8
```

#### Parameter of Batch normalization
```python
EPSILON = 0
```

#### Number of labels
```python
NUM_LABELS = 10
```

#### Unit dynamic energy
```python
## Computation
ENERGY_POPCOUNT = 0.00054075 # popcount
ENERGY_XOR = 2.7124E-07 #xor
ENERGY_OR = 2.25811E-06 # OR logic
ENERGY_BNA = 0.000437255 # batch normalization and activation
ENERGY_COMPARISON = 5.9293E-05 # Comparison

## Data movement
ENERGY_DM_READ_BUFFER_IW = 2.21987E-06 # read from buffer
ENERGY_DM_LOAD_PE = 4.08203E-07 # load data to PE
ENERGY_DM_REAM_BUFFER_BIAS = 5.01043E-06 # read from bias buffer
ENERGY_DM_LOAD_CONTROL = 5.9393E-05 # load control
```

#### Unit leakage energy
```python
# Computation
LEAK_POPCOUNT = 500 # popcount
LEAK_XOR = 10.25390625 # xor
LEAK_OR = 24.63866016 # OR logic
LEAK_BNA = 791.7637969 # batch normalization and activation
LEAK_COMPARISON = 509.1364333 # comparison

## Data movement
LEAK_DM_READ_BUFFER_IW = 23.25170117 # read from buffer
LEAK_DM_REAM_BUFFER_BIAS = 14.56696875 # read from bias
LEAK_DM_LOAD_CONTROL = 31.4090625 # load control
```

#### Clock period
```python
CLOCK_PERIOD = 0.000000005
```

### ◼️ Run Example
#### Example 1.
Weights and bias are defined with random values
```python
python Example_simple.py
```
#### Example 2.
Program includes loading a pre-trained model for cifar10
```python
python main.py
```

## 🟨 Citing LAXOR accelerator and simulator

If you use LAXOR, please cite the following paper:
```
Paper information
```

## 🟨 Licence

[MIT license](https://en.wikipedia.org/wiki/MIT_License).

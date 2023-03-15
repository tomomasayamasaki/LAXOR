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
    

- Pre-trained_model

    Pre-trained model of Binary CNN for CIFAR-10, which the accuracy is 85.25%, the total enegy is 3.82$uJ$, and the model size is 0.51$MB$.
    
- [Example_simple.py](https://github.com/tomomasayamasaki/LAXOR/blob/main/Example_simple.py)


## 🟨 Licence

[MIT license](https://en.wikipedia.org/wiki/MIT_License).

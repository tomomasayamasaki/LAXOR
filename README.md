<a href="https://istd.sutd.edu.sg/people/phd-students/tomomasa-yamasaki">
    <img src="https://github.com/tomomasayamasaki/LAXOR/blob/main/README/logo.png" alt="Tomo logo" title="Tomo" align="right" height="110" />
</a>

# LAXOR: A Scalable BNN Accelerator with Latch-XOR Logic for Local Computing
  
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![](https://img.shields.io/github/downloads/tomomasayamasaki/LAXOR/total)
![](https://img.shields.io/github/repo-size/tomomasayamasaki/LAXOR)
![](https://img.shields.io/github/commit-activity/y/tomomasayamasaki/LAXOR)
![](https://img.shields.io/github/last-commit/tomomasayamasaki/LAXOR)
![](https://img.shields.io/github/languages/count/tomomasayamasaki/LAXOR)

![gif](https://github.com/tomomasayamasaki/LAXOR/blob/main/README/LAXOR.gif)

## Contents
- [Introduction](#introduction)
- [Licence](#licence)

## Introduction
We design a python-based simulator for the proposed LAXOR accelerator. The purpose of the simulator is to
- (1) map and verify the functionality of a BNN model onto the proposed architecture
- (2) generate application-specific, cycle-accurate results (e.g., latency, energy, utilization, etc.) for design analysis.

The LAXOR simulator consists of a front-end tool, Areca, and a back-end tool,Bits-Island. Areca interfaces with the pre-trained model and user configurations before generating the data stream in a format tailored to the accelerator. Bits-Island replicates the LAXOR architecture, maps the data stream onto different PEs, and simulates the functionality layer by layer. Eventually the tool-chain reports the mapping results, layer output, and critical design metrics by harnessing embedded cycle count, latency and energy models. To ensure accurate energy estimation, latency and energy per atomic hardware operation such as single XOR gate, buffer read, weight loading, are provided to Areca using Cadence Spectre gate-level and post layout simulations.

<p align="center"><img width=80% src="https://github.com/tomomasayamasaki/LAXOR/blob/main/README/fig1.png"></p>

## Licence

[MIT license](https://en.wikipedia.org/wiki/MIT_License).

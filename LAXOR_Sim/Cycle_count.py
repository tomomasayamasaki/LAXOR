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


import LAXOR_Sim.Config as config
import numpy as np
from LAXOR_Sim.Performance import performance
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick

class cycle_count:
    CLK_LOAD_ROW_ADDRESS = 1
    CLK_ACTIVE_TG = 1
    CLK_DEACTIVE_TG = 1
    CLK_LOAD_BIAS_DONE = 1
    CLK_LOAD_INPUT_DONE = 1
    CLK_BIAS_IN = 1
    CLK_PASS_RESULT_POPCOUNT = 1
    CLK_BN = 1
    CLK_FCEN = 1
    CLK_SIGN = 1
    CLK_ACTIVE_MAXPOOLING = 1
    CLK_RUN_MAXPOOLING = 1
    CLK_LOAD_4PSUM = 1
    CLK_ADD_4PSUM = 1
    CLK_SUM_OUT = 1
    CLK_CLASSIFY = 6
    CLK_RUN_PE = 1 # 20 Jan 2023 updated 8 -> 1
    CLK_RUN_XOR = 1

    def __init__(self):
        self.CLK_LOAD_WEIGHT_BUFFER = np.ceil(config.BUFFERSIZE_WEIGHTS / config.PINS_IW)
        self.CLK_LOAD_INPUT_BUFFER = np.ceil(config.BUFFERSIZE_INPUT / config.PINS_IW)
        self.CLK_LOAD_BIAS_BUFFER = config.BUFFERSIZE_BIAS
        self.total_weights = 0
        self.total_input = 0
        self.total_bias = 0
        self.total_PE = 0
        self.total_maxp = 0
        self.total_comp = 0
        self.total_BNA = 0
        self.pf = performance()
        self.logfile_name = config.LOG_FILE
        if not os.path.exists('Sim Result'):
            os.makedirs('Sim Result')
        self.logfile_name = './Sim Result/'+self.logfile_name
        self.cycle_value = list()
        self.cycle_layer = list()


    def get_cycle_count(self, layer, weights_height, input_height, input_channel, T, C=1, c_flag=False):

        cycle = 0
        if layer == 'Binary_Conv2D':
            # weights
            load_cycle_weights = 0
            load_cycle_weights += np.ceil(((weights_height*config.BIT_SIZE_PE)/config.BUFFERSIZE_WEIGHTS)) * self.CLK_LOAD_WEIGHT_BUFFER
            load_cycle_weights += weights_height * (cycle_count.CLK_ACTIVE_TG+cycle_count.CLK_DEACTIVE_TG)
            load_cycle_weights += weights_height * cycle_count.CLK_LOAD_ROW_ADDRESS
            load_cycle_weights = np.ceil(load_cycle_weights/512)
            cycle += load_cycle_weights
            self.total_weights += load_cycle_weights

            # bias
            """
            load_cycle_bias = 0
            load_cycle_bias += self.CLK_LOAD_BIAS_BUFFER * config.PENUMS * T
            cycle += load_cycle_bias
            self.total_bias += load_cycle_bias
            """
            self.total_bias = 0 # bias is loaded to PE with weights

            # input
            load_cycle_input = 0
            load_cycle_input += C * T * int(np.ceil(((input_height*config.BIT_SIZE_PE)/config.BUFFERSIZE_INPUT))) * self.CLK_LOAD_INPUT_BUFFER
            load_cycle_input = np.ceil(load_cycle_input/512)
            cycle += load_cycle_input
            self.total_input += load_cycle_input

            # PE
            Num_Operations = input_height
            PE_cycle = 0
            PE_cycle += C * T * cycle_count.CLK_RUN_PE * Num_Operations
            cycle += PE_cycle
            self.total_PE += PE_cycle

            # BNA
            BNA_cycle = 0
            BNA_cycle += C * T * (cycle_count.CLK_BIAS_IN + cycle_count.CLK_BN +  cycle_count.CLK_SIGN)
            cycle += BNA_cycle
            self.total_BNA += BNA_cycle

            f = open(self.logfile_name, 'a')
            txt ='---------------------------------------------\n'
            txt +=  '*** CYCLE COUNT *** \n'
            txt += ' TOTAL for this layer      : {0:9.0f} cycles\n'.format(int(cycle))
            txt += '    * Load weights         : {0:9.2f} % ({1:9.2f})\n'.format(100*load_cycle_weights/cycle, load_cycle_weights)
            txt += '    * Load input           : {0:9.2f} % ({1:9.2f})\n'.format(100*load_cycle_input/cycle, load_cycle_input)
            txt += '    * Run PE               : {0:9.2f} % ({1:9.2f})\n'.format(100*PE_cycle/cycle, PE_cycle)
            txt += '    * Run BNA              : {0:9.2f} % ({1:9.2f})'.format(100*BNA_cycle/cycle, BNA_cycle)
            print(txt)
            f.write('\n')
            f.write(txt)
            f.close()
            layer = 'Conv'

            # Leakage


        elif layer == 'MaxPooling':
            # Load input to buffer
            load_cycle_input = 0
            load_cycle_input += np.ceil(input_height*input_height*input_channel/config.BUFFERSIZE_INPUT) * self.CLK_LOAD_INPUT_BUFFER
            load_cycle_input = np.ceil(load_cycle_input/512)
            cycle += load_cycle_input
            self.total_input += load_cycle_input

            # Activate maxpooling
            # Maxpooling
            Run_maxpool = 0
            Run_maxpool += T * (cycle_count.CLK_ACTIVE_MAXPOOLING+cycle_count.CLK_RUN_MAXPOOLING)
            cycle += Run_maxpool
            self.total_maxp += Run_maxpool

            f = open(self.logfile_name, 'a')
            txt ='---------------------------------------------\n'
            txt +=  '*** CYCLE COUNT ***\n'
            txt += ' TOTAL for this layer      : {0:9.0f} cycles\n'.format(int(cycle))
            txt += '    * Load input           : {0:9.2f} % ({1:9.2f})\n'.format(100*load_cycle_input/cycle, load_cycle_input)
            txt += '    * Run MaxPool          : {0:9.2f} % ({1:9.2f})'.format(100*Run_maxpool/cycle, Run_maxpool)
            print(txt)
            f.write('\n')
            f.write(txt)
            f.close()
            layer = 'MP'



        elif layer == 'Binary_FullyConnected':
            # weights
            load_cycle_weights = 0
            load_cycle_weights += np.ceil(((weights_height*config.BIT_SIZE_PE)/config.BUFFERSIZE_WEIGHTS)) * self.CLK_LOAD_WEIGHT_BUFFER
            load_cycle_weights += weights_height * (cycle_count.CLK_ACTIVE_TG+cycle_count.CLK_DEACTIVE_TG)
            load_cycle_weights += weights_height * cycle_count.CLK_LOAD_ROW_ADDRESS
            load_cycle_weights = np.ceil(load_cycle_weights/512)
            cycle += load_cycle_weights
            self.total_weights += load_cycle_weights


            # bias
            """
            load_cycle_bias = 0
            load_cycle_bias += self.CLK_LOAD_BIAS_BUFFER * config.PENUMS * T
            cycle += load_cycle_bias
            self.total_bias += load_cycle_bias
            """
            self.total_bias = 0 # bias is loaded to PE with weights


            # input
            load_cycle_input = 0
            load_cycle_input += C * T * np.ceil(input_height*config.BIT_SIZE_PE/config.BUFFERSIZE_INPUT)* self.CLK_LOAD_INPUT_BUFFER
            load_cycle_input = np.ceil(load_cycle_input/512)
            cycle += load_cycle_input
            self.total_input += load_cycle_input

            # PE
            Num_Operations = np.ceil(weights_height/config.PENUMS)
            PE_cycle = 0
            PE_cycle += C * T * cycle_count.CLK_RUN_PE * Num_Operations
            cycle += PE_cycle
            self.total_PE += PE_cycle

            # BNA
            BNA_cycle = 0
            if not c_flag:
                BNA_cycle += C * T * (cycle_count.CLK_BIAS_IN + cycle_count.CLK_BN +  cycle_count.CLK_SIGN)
                cycle += BNA_cycle
                self.total_BNA += BNA_cycle

            # OUTPUT
            comp_cycle = 0
            if c_flag:
                comp_cycle += cycle_count.CLK_LOAD_4PSUM+cycle_count.CLK_ADD_4PSUM
                comp_cycle += cycle_count.CLK_SUM_OUT
                comp_cycle += cycle_count.CLK_CLASSIFY
                cycle += comp_cycle
                self.total_comp += comp_cycle


            f = open(self.logfile_name, 'a')
            txt ='---------------------------------------------\n'
            txt += '*** CYCLE COUNT ***\n'
            txt +=' TOTAL for this layer      : {0:9.0f} cycles\n'.format(int(cycle))
            txt +='    * Load weights         : {0:9.2f} % ({1:9.2f})\n'.format(100*load_cycle_weights/cycle, load_cycle_weights)
            txt +='    * Load input           : {0:9.2f} % ({1:9.2f})\n'.format(100*load_cycle_input/cycle, load_cycle_input)
            txt +='    * Run PE               : {0:9.2f} % ({1:9.2f})\n'.format(100*PE_cycle/cycle, PE_cycle)
            if c_flag:
                txt +='    * Comparison           : {0:9.2f} % ({1:9.2f})'.format(100*comp_cycle/cycle, comp_cycle)
            else:
                txt +='    * Run BNA           : {0:9.2f} % ({1:9.2f})'.format(100*BNA_cycle/cycle, BNA_cycle)
            print(txt)
            f.write('\n')
            f.write(txt)
            f.close()
            layer = 'FC'



        else:
            print('Error@cycle_count.py: your layer name is wrong. Please use Binary_Conv2D, Binary_FullyConnected, or MaxPooling')
            exit()

        self.cycle_value.append(int(cycle))
        self.cycle_layer.append(layer)

        if c_flag:
            total = self.total_weights + self.total_input + self.total_bias + self.total_PE + self.total_maxp + self.total_comp + self.total_BNA
            latency = self.pf.get_performace(total)
            f = open(self.logfile_name, 'a')
            txt = '\n'
            txt +=' TOTAL                     : {0:9.0f} cycles\n'.format(int(total))
            txt +='    * Load weights         : {0:9.2f} %\n'.format(100*self.total_weights/total)
            txt +='    * Load input           : {0:9.2f} %\n'.format(100*self.total_input/total)
            txt +='    * Run PE               : {0:9.2f} % ({1:9.2f}cycle)\n'.format(100*self.total_PE/total,self.total_PE)
            txt +='    * Run MaxPool          : {0:9.2f} %\n'.format(100*self.total_maxp/total)
            txt +='    * Comparison           : {0:9.2f} %\n'.format(100*self.total_comp/total)
            txt +='---------------------------------------------\n'
            txt +='*** PERFORMANCE ***\n'
            txt +=' TOTAL Latency             : {0:9.4f} ms\n'.format(latency)
            txt +='---------------------------------------------'
            print(txt)
            f.write('\n')
            f.write(txt)
            f.close()



            name = config.LOG_FILE.split('.')
            fig, ax = plt.subplots()
            ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="y", scilimits=(6,6))
            x_position = np.arange(len(self.cycle_layer))
            bar =plt.bar(x_position, self.cycle_value, color="gray", tick_label=self.cycle_layer,edgecolor='k')
            plt.title('Cycle Count')
            plt.xlabel('Layer')
            plt.ylabel('Cycle')
            plt.savefig("./Sim Result/cycle_count_"+name[0]+".eps")


            np.savetxt("./Sim Result/cycle_count_"+name[0]+".csv", np.array(self.cycle_value), delimiter=",")
            np.savetxt("./Sim Result/latency_"+name[0]+".csv", np.array([latency]), delimiter=",")

            return int(total), int(self.total_PE)

        return 0, 0

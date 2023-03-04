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
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ptick

class energy:
    def __init__(self):
        self.total_energy_comp = 0
        self.total_energy_datamove = 0
        self.total_energy = 0
        self.logfile_name = config.LOG_FILE
        if not os.path.exists('Sim Result'):
            os.makedirs('Sim Result')
        self.logfile_name = './Sim Result/'+self.logfile_name
        self.number = 1
        self.energy_comp = list()
        self.energy_dm = list()
        self.energy_sum = list()
        self.energy_layer = list()
        self.leakage = 0
        self.comparison_input = 0

    def get_energy(self, layer_type, weights_height, input_height, input_channel, T, datasize_output, C=1, c_flag=False, Total_cycle=0, PE_cycle=0):
        energy_comp = 0
        energy_datamove = 0
        energy = 0
        if layer_type == 'Binary_Conv2D':
            num_operation = weights_height*input_height
            # computation
            ## XOR
            energy_comp += num_operation*config.BIT_SIZE_PE*config.ENERGY_XOR
            ## POPCOUNT
            energy_comp += num_operation*np.ceil(config.BIT_SIZE_PE/128)*config.ENERGY_POPCOUNT
            ## BN and Sign
            energy_comp += num_operation*config.ENERGY_BNA

            self.total_energy_comp += energy_comp


            # data movement
            ## Weights Buffer (read)
            energy_datamove += weights_height*config.BIT_SIZE_PE*config.ENERGY_DM_READ_BUFFER_IW
            ## PE load weight (TG) control
            energy_datamove += T*config.PENUMS*config.ENERGY_DM_LOAD_CONTROL
            ## Input buffer (read)
            energy_datamove += input_height*config.BIT_SIZE_PE*config.ENERGY_DM_READ_BUFFER_IW
            ## Weight loading to XOR
            energy_datamove += weights_height*config.BIT_SIZE_PE*config.ENERGY_DM_LOAD_PE

            self.total_energy_datamove += energy_datamove

            # total
            energy = energy_comp + energy_datamove
            self.total_energy += energy
            layer = 'Conv'


        elif layer_type == 'Binary_FullyConnected':
            num_operation = weights_height
            # computation
            ## XOR
            energy_comp += num_operation*config.BIT_SIZE_PE*config.ENERGY_XOR
            ## POPCOUNT
            energy_comp += num_operation*np.ceil(config.BIT_SIZE_PE/128)*config.ENERGY_POPCOUNT
            if not c_flag:
                ## BN and Sign
                energy_comp += num_operation*config.ENERGY_BNA
            else:
                # FC (N accumulators)
                self.comparison_input =  num_operation/config.NUM_LABELS
                energy_comp += (self.comparison_input-1)*config.NUM_LABELS*config.ENERGY_COMPARISON

            self.total_energy_comp += energy_comp

            # data movement
            ## Weights Buffer (read)
            energy_datamove += weights_height*config.BIT_SIZE_PE*config.ENERGY_DM_READ_BUFFER_IW
            ## PE load weight (TG) control
            energy_datamove += T*config.PENUMS*config.ENERGY_DM_LOAD_CONTROL
            ## Input buffer (read)
            energy_datamove += input_height*config.BIT_SIZE_PE*config.ENERGY_DM_READ_BUFFER_IW
            ## Weight loading to XOR
            energy_datamove += weights_height*config.BIT_SIZE_PE*config.ENERGY_DM_LOAD_PE

            self.total_energy_datamove += energy_datamove
            
            # total
            energy = energy_comp + energy_datamove
            self.total_energy += energy
            layer = 'FC'


        elif layer_type == 'MaxPooling':
            outsize = weights_height
            # computation
            energy_comp += outsize*outsize*input_channel*config.ENERGY_OR

            self.total_energy_comp += energy_comp

            # data movement
            energy_datamove += input_height*input_height*input_channel*config.ENERGY_DM_READ_BUFFER_IW

            self.total_energy_datamove += energy_datamove

            # total
            energy = energy_comp + energy_datamove
            self.total_energy += energy
            layer = 'MP'


        else:
            print('Error@energy.py: your layer type is wrong. Please use Binary_Conv2D, Binary_FullyConnected, or MaxPooling')
            exit()

        self.energy_comp.append(energy_comp)
        self.energy_dm.append(energy_datamove)
        self.energy_sum.append(energy)
        #self.energy_layer.append(layer+'_'+str(self.number))
        self.energy_layer.append(layer)
        self.number += 1
        f = open(self.logfile_name, 'a')
        txt =  '*** ESTIMATED ENERGY  *** \n'
        txt += ' Energy for this layer     : {0:9} nJ\n'.format(round(energy, 4))
        txt += '    * computation energy   : {0:9} nJ\n'.format(round(energy_comp, 4))
        txt += '    * data movement energy : {0:9} nJ'.format(round(energy_datamove, 4))
        print(txt)
        f.write('\n')
        f.write(txt)
        f.close()

        if c_flag and Total_cycle > 0 and PE_cycle > 0:
            self.get_leakage(Total_cycle, PE_cycle)
            f = open(self.logfile_name, 'a')
            txt ='---------------------------------------------\n\n\n'
            txt +='ENERGY CONSUMPTION FOR YOUR MODEL\n'
            txt +='+++++++++++++++++++++++++++++++++++++++++++++++++++\n'
            txt +='+ Dynamic Energy            : {0:9} uJ        +\n'.format(round(self.total_energy/1000, 6))
            txt +='+    * computation energy   :   {0:9} uJ      +\n'.format(round(self.total_energy_comp/1000, 6))
            txt +='+    * data movement energy :   {0:9} uJ      +\n'.format(round(self.total_energy_datamove/1000, 6))
            txt +='+ leakage Energy            : {0:9} uJ        +\n'.format(round(self.leakage/1000, 6))
            txt +='+ Total Energy              : {0:9} uJ        +\n'.format(round(self.leakage/1000+self.total_energy/1000, 6))
            txt +='+                                                 +\n'
            txt +='+ PERCENTAGE                                      +\n'
            txt +='+    * computation energy   : {0:9.2f} %         +\n'.format(100*self.total_energy_comp/self.total_energy)
            txt +='+    * data movement energy : {0:9.2f} %         +\n'.format(100*self.total_energy_datamove/self.total_energy)
            txt += '+++++++++++++++++++++++++++++++++++++++++++++++++++\n'

            print(txt)
            f.write('\n')
            f.write(txt)
            f.close()

            #self.energy_comp.append(self.total_energy_comp)
            #self.energy_dm.append(self.total_energy_datamove)
            #self.energy_layer.append('Total')

            name = config.LOG_FILE.split('.')
            fig, ax = plt.subplots()
            ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="y", scilimits=(9,9))
            x_position = np.arange(len(self.energy_layer))
            dm_bar = plt.bar(x_position, self.energy_dm, color="green", tick_label=self.energy_layer)
            comp_bar =plt.bar(x_position, self.energy_comp, bottom=self.energy_dm, color="orange", tick_label=self.energy_layer)
            plt.legend((dm_bar[0], comp_bar[0]), ("data movement energy", "computational energy"))
            plt.title('Energy Consumption')
            plt.xlabel('Layer')
            plt.ylabel('Energy [nJ]')
            plt.savefig("./Sim Result/energy_color_"+name[0]+".eps")


            fig, ax = plt.subplots()
            ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="y", scilimits=(9,9))
            x_position = np.arange(len(self.energy_layer))
            dm_bar = plt.bar(x_position, self.energy_dm, color="white", tick_label=self.energy_layer, hatch="//",edgecolor='k')
            comp_bar =plt.bar(x_position, self.energy_comp, bottom=self.energy_dm, color="gray", tick_label=self.energy_layer,edgecolor='k')
            plt.legend((dm_bar[0], comp_bar[0]), ("data movement energy", "computational energy"))
            plt.title('Energy Consumption')
            plt.xlabel('Layer')
            plt.ylabel('Energy [nJ]')
            plt.savefig("./Sim Result/energy_gray_"+name[0]+".eps")

            fig, ax = plt.subplots()
            ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
            #ax.ticklabel_format(style="sci", axis="y", scilimits=(9,9))
            x_position = np.arange(len(self.energy_layer))
            comp_bar =plt.bar(x_position, self.energy_sum, color="gray", tick_label=self.energy_layer,edgecolor='k')
            plt.title('Energy Consumption')
            plt.xlabel('Layer')
            plt.ylabel('Energy [J]')
            plt.savefig("./Sim Result/energy_total_"+name[0]+".eps")

            np.savetxt("./Sim Result/energy_consumption_"+name[0]+".csv", np.array(self.energy_sum), delimiter=",")

    #############################################
    #
    #   LEAKAGE
    #
    #
    #############################################
    def get_leakage(self, total_cycle, PE_cycle):
        other_cycle = total_cycle - PE_cycle

        # power
        leakpower_popcount = config.LEAK_POPCOUNT*np.ceil(config.BIT_SIZE_PE/128)*config.PENUMS
        leakpower_xor = config.LEAK_XOR*config.BIT_SIZE_PE*config.PENUMS
        leakpower_or = config.LEAK_OR*config.ORNUMS
        leakpower_bna = config.LEAK_BNA*config.PENUMS
        leakpower_comparison = config.LEAK_COMPARISON*(self.comparison_input-1)*config.NUM_LABELS
        leakpower_buffer_iw = config.LEAK_DM_READ_BUFFER_IW*config.BUFFERSIZE_INPUT
        leakpower_buffer_bias = config.LEAK_DM_REAM_BUFFER_BIAS*config.PENUMS*config.BUFFERSIZE_BIAS
        leakpower_load_control = config.LEAK_DM_LOAD_CONTROL*config.PENUMS

        leakage_power_other = leakpower_or+leakpower_bna+leakpower_comparison+leakpower_buffer_iw+leakpower_buffer_bias+leakpower_load_control
        leakage_power_pe = leakpower_popcount+leakpower_xor
        leakage_power_total = leakage_power_other+leakage_power_pe
        
        self.leakage = leakage_power_total*total_cycle*config.CLOCK_PERIOD

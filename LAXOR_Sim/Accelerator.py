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

# import
import numpy as np
from LAXOR_Sim.PE import PE
from LAXOR_Sim.Cycle_count import cycle_count
from LAXOR_Sim.Energy import energy
import LAXOR_Sim.Config as config
import LAXOR_Sim.Tool as tool
import os
import datetime



class Accelerator_Engine:


    def __init__(self):
        # init cycle count

        # PE
        self.PE_Num = config.PENUMS
        self.bitsize_pe = config.BIT_SIZE_PE
        self.BufferSize_W = config.BUFFERSIZE_WEIGHTS
        self.BufferSize_I = config.BUFFERSIZE_INPUT
        self.OR_numbers = config.ORNUMS
        self.OR_bitwidth = config.ORBITWIDTH

        self.PE_util = 0
        self.PE_total = 0

        self.cc = cycle_count()
        self.energy = energy()

        self.__InitPEs__()

        self.logfile_name = config.LOG_FILE
        if not os.path.exists('Sim Result'):
            os.makedirs('Sim Result')
        self.logfile_name = './Sim Result/'+self.logfile_name
        if os.path.exists(self.logfile_name):
            f = open(self.logfile_name, 'a')
            f.write('\n\n\n')
            f.write('NEXT SIMULATOR RUNNING RESULT\n##################################################################\n')
            f.write('##################################################################\n')
            f.write('##################################################################\n')
        else:
            f = open(self.logfile_name, 'w')

        intro = '\n\n'
        intro +='               ██╗      █████╗ ██╗  ██╗ ██████╗ ██████╗                    \n'
        intro +='               ██║     ██╔══██╗╚██╗██╔╝██╔═══██╗██╔══██╗                   \n'
        intro +='               ██║     ███████║ ╚███╔╝ ██║   ██║██████╔╝                   \n'
        intro +='               ██║     ██╔══██║ ██╔██╗ ██║   ██║██╔══██╗                   \n'
        intro +='               ███████╗██║  ██║██╔╝ ██╗╚██████╔╝██║  ██║                   \n'
        intro +='               ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝                   \n'
        intro +='███████╗██╗███╗   ███╗██╗   ██╗██╗      █████╗ ████████╗ ██████╗ ██████╗ \n'
        intro +='██╔════╝██║████╗ ████║██║   ██║██║     ██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗\n'
        intro +='███████╗██║██╔████╔██║██║   ██║██║     ███████║   ██║   ██║   ██║██████╔╝\n'
        intro +='╚════██║██║██║╚██╔╝██║██║   ██║██║     ██╔══██║   ██║   ██║   ██║██╔══██╗\n'
        intro +='███████║██║██║ ╚═╝ ██║╚██████╔╝███████╗██║  ██║   ██║   ╚██████╔╝██║  ██║\n'
        intro +='╚══════╝╚═╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝\n'
        print(intro)
        f.write(intro)


        # Show Configuration
        conf = 'CONFIGURATION\n'
        conf += 'LOG FILE NAME: {}\n'.format(config.LOG_FILE)
        dt_now = datetime.datetime.now()
        conf += 'RUN DATE: {}\n'.format(dt_now.strftime('%Y-%m-%d %H:%M:%S'))
        conf += '+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'
        conf += '+ The No. of PEs         : {0:4} [PE]\n'.format(config.PENUMS)
        conf += '+ The Size of PE         : {0:4} [bits]\n'.format(config.BIT_SIZE_PE)
        conf += '+ The No. of I/O         : {0:4} [pins]\n'.format(config.PINS_IW)
        conf += '+ Buffer size for INPUT  : {0:4} [bits]\n'.format(config.BUFFERSIZE_INPUT)
        conf += '+ Buffer size for WEIGHT : {0:4} [bits]\n'.format(config.BUFFERSIZE_WEIGHTS)
        conf += '+ Buffer size for BIAS   : {0:4} [bits]\n'.format(config.BUFFERSIZE_BIAS)
        conf += '+ EPSILON for BatchNorm  : {0:4}\n'.format(config.EPSILON)
        conf += '+ The No. of labels      : {0:4} [labels]\n'.format(config.NUM_LABELS)
        conf += '+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'
        print(conf)
        f.write('\n')
        f.write(conf)

        f.close()




    def __InitPEs__(self):
        """
            build PEs for XNOR
        """
        self.PEs = list()
        for x in range(self.PE_Num):
            self.PEs.append(PE())



    def _DeliverWeights(self, subweights, No):
        self.PEs[No].SetXNORWeights(subweights)



    def _DeliverInput(self, subinput, No):
        self.PEs[No].SetXNORInput(subinput)



    ##################################################################################################
    # API
    #    Binary Convolution Layer
    #    name: Binary_Conv2D
    #    parameters:
    #        1. input
    #        2. weights
    #        3. bias
    #        4. datasize_output
    #        5. mapping
    #        6. unit_row
    #        7. out_size
    ##################################################################################################
    def Binary_Conv2D(self, input, weights, bias, datasize_output, mapping, unit_row, out_size):
        count_xor = 0
        count_pop = 0
        """
            Compute Convolution
        """
        self.layer_type = 'Binary_Conv2D'
        self.mapping = mapping # 'Single_PE', 'Part_PE', 'Multi_PE'
        self.unit_Row = unit_row
        self.l = out_size
        self.F = int(datasize_output/(self.l*self.l))

        fnum, w_width = weights.shape
        inputrows, i_width = input.shape

        # cycles
        self.T = int(np.ceil(fnum/self.PE_Num))
        self.L = fnum
        self.P = self.PE_Num
        self.Npe = 1
        if self.mapping == 'Multi_PE':
            self.Npe = int(np.ceil(self.unit_Row/self.bitsize_pe))
            self.L = int(fnum/self.Npe) # the total of Convolution rows

            # If one convolution needs more PEs than the number of PEs
            # It is software-based solution
            if self.PE_Num < self.Npe:
                self.PE_Num = self.Npe
                self.__InitPEs__()

            self.P = int(self.PE_Num/self.Npe) # convolution for 1 cycle
            self.T = int(np.ceil(self.L/self.P)) # cycles which all convlutions need

        # classify flag
        c_flag = False
        if datasize_output == config.NUM_LABELS:
            c_flag = True

        # output
        self.output = np.zeros((1,self.F,self.l,self.l))

        # Analize used PE numbers
        TPE = list()

        # mapping uses single PE for one convolution
        if self.mapping == 'Single_PE' or self.mapping == 'Part_PE':
            # Compute
            remaider =  self.L % self.P

            for t in range(self.T):
                N = self.P
                if t == self.T-1:
                    if remaider != 0:
                        N = remaider
                if self.L < self.P:
                    N = self.L
                TPE.append(N*self.Npe)

                kh = 0
                kw = 0
                # Weights
                for no in range(N):
                    # load weights to buffer
                    ind = t * self.P + no
                    row_weight = weights[ind,:]

                    # set weights to PE
                    self._DeliverWeights(row_weight,no)



                # Bias
                for no in range(N):
                    ind = t * self.P + no
                    # load bias
                    self.PEs[no].SetPopArray(bias[ind],'bias')


                # input
                for input_ind in range(inputrows):
                    # load input to buffer
                    row_input = input[input_ind,:]

                    # set input to PE
                    for no in range(N):
                        self._DeliverInput(row_input,no)

                    # XNOR
                    for no in range(N):
                        # run XNOR
                        self.PEs[no].RunXNOR()
                        count_xor += 1


                    # popcount
                    for no in range(N):
                        ind = t * self.P + no

                        # run popcount
                        POPCout = self.PEs[no].RunPopcount()
                        count_pop += 1

                        # pass the result to BRAM
                        self.output[0,ind,kh,kw] = POPCout


                    kw += 1
                    if kw == self.l:
                        kw = 0
                        kh += 1

        # mapping uses multi PE for one convolution with this condition: self.Npe <= self.PE_Num (self.P > 0)
        elif self.mapping == 'Multi_PE':
            remaider =  self.L % self.P

            for t in range(self.T):

                N = self.P
                if t == self.T-1:
                    if remaider != 0:
                        N = remaider
                if self.L < self.P:
                    N = self.L
                TPE.append(N*self.Npe)

                # Weights
                no = 0
                for p in range(N):
                    for npe in range(self.Npe):
                        # load weights to buffer
                        ind = t*self.P*self.Npe + p*self.Npe + npe
                        row_weight = weights[ind,:]

                        # set weights to PE
                        self._DeliverWeights(row_weight,no)

                        no += 1

                # Bias
                no = 0
                for p in range(N):
                    ind = t*self.P + p
                    for npe in range(self.Npe):
                        if npe == 0:
                            self.PEs[no].SetPopArray(bias[ind],'bias')
                        else:
                            self.PEs[no].SetPopArray(0,'bias')
                        no += 1

                # Input
                kh = 0
                kw = 0
                for i in range(int(inputrows/self.Npe)):

                    # load input to buffer
                    for npe in range(self.Npe):
                        ind = i*self.Npe + npe
                        row_input = input[ind,:]

                        # set input to PE
                        for p in range(N):
                            no = p*self.Npe + npe
                            self._DeliverInput(row_input,no)

                    # XNOR
                    for no in range(N*self.Npe):
                        self.PEs[no].RunXNOR()
                        count_xor += 1

                    # Popcount
                    for p in range(N):
                        fifo = 0
                        ind = t*self.P + p
                        for npe in range(self.Npe):
                            no = p*self.Npe + npe
                            POPCout = self.PEs[no].RunPopcount()
                            count_pop += 1
                            fifo += POPCout
                        self.output[0,ind,kh,kw] = fifo



                    kw += 1
                    if kw == self.l:
                        kw = 0
                        kh += 1



        # Show used PE number
        f = open(self.logfile_name, 'a')
        txt = 'Mapping: {} (1 PE for one convolution)\n'.format(self.mapping)
        if self.mapping == 'Multi_PE':
            txt = 'Mapping: {} ({} PEs for one convolution)\n'.format(self.mapping, self.Npe)
        txt += 'PE Utilization Rate:  {0:9.2f} %\n'.format(100*sum(TPE)/(config.PENUMS*self.T*np.ceil(self.Npe/config.PENUMS)))
        no = 1
        txt += 'Computation Round\n'
        for numpe in TPE:
            for c in range(int(np.ceil(self.Npe/config.PENUMS))):
                if numpe > config.PENUMS:
                    txt += '    {0:<5}:{1:>4}/{2:<4}PEs\n'.format(no,config.PENUMS,config.PENUMS)
                    self.PE_util += config.PENUMS
                    self.PE_total += config.PENUMS
                    numpe = numpe - config.PENUMS
                else:
                    txt += '    {0:<5}:{1:>4}/{2:<4}PEs\n'.format(no,numpe,config.PENUMS)
                    self.PE_util += numpe
                    self.PE_total += config.PENUMS
                no += 1

        print(txt)
        f.write(txt)
        f.close()

        total_cycle, PE_cycle = self.cc.get_cycle_count(self.layer_type, fnum, inputrows, 0, self.T, int(np.ceil(self.Npe/config.PENUMS)),c_flag)
        self.energy.get_energy(self.layer_type, fnum, inputrows, 0, self.T, datasize_output, int(np.ceil(self.Npe/config.PENUMS)), c_flag, total_cycle, PE_cycle)


        f = open(self.logfile_name, 'a')
        txt = '*** OUTPUT INFO ***\n'
        txt +=' Shape                     : {}\n'.format(self.output.shape)
        txt +='---------------------------------------------\n\n'
        print(txt)
        f.write(txt)
        f.close()


        if c_flag == True:
            self.output = self.output.reshape(config.NUM_LABELS)
            self.output = np.argmax(self.output)


        return self.output


    ##################################################################################################
    # API
    #    Binary Fully Connected Layer
    #    name: Binary_FullyConnected
    #    parameters:
    #        1. input
    #        2. weights
    #        3. bias
    #        4. mapping
    #        5. unit_row
    ##################################################################################################
    def Binary_FullyConnected(self, input, weights, bias, mapping, unit_row):
        count_xor = 0
        count_pop = 0
        """
            Fully Connected layer

            *parameters*
            input: (1,256,h,h) numpy
            weights: (w, 10) numpy
            bias: (10) numpy
        """
        self.layer_type = 'Binary_FullyConnected'
        self.mapping = mapping
        self.unit_Row = unit_row

        # Compute Fully Connected layers
        w_height, w_width = weights.shape
        i_height, i_width = input.shape

        self.T = int(np.ceil(w_height/self.PE_Num))
        self.L = w_height
        self.P = self.PE_Num
        self.Npe = 1
        if self.mapping == 'Multi_PE':
            self.Npe = int(np.ceil(self.unit_Row/self.bitsize_pe))

            # If one convolution needs more PEs than the number of PEs
            # It is software-based solution
            if self.PE_Num < self.Npe:
                self.PE_Num = self.Npe
                self.__InitPEs__()

            self.L = int(w_height/self.Npe)
            self.P = int(self.PE_Num/self.Npe)
            self.T = int(np.ceil(self.L/self.P))

        # output
        self.output = np.zeros((1,self.L))

        # Analize used PE numbers
        TPE = list()

        # classify flag
        c_flag = False
        if self.L == config.NUM_LABELS:
            c_flag = True

        # mapping uses single PE
        if self.mapping == 'Single_PE' or self.mapping == 'Part_PE':
            # Compute
            remaider =  self.L % self.P

            for t in range(self.T):
                N = self.P
                if t == self.T-1:
                    if remaider != 0:
                        N = remaider
                if self.L < self.P:
                    N = self.L
                TPE.append(N*self.Npe)

                # weights
                for no in range(N):
                    ind = t*self.P + no
                    row_weight = weights[ind,:]

                    # set weights to PE
                    self._DeliverWeights(row_weight,no)

                # Bias
                for no in range(N):
                    ind = t*self.P+ no
                    # load bias
                    self.PEs[no].SetPopArray(bias[ind],'bias')

                # Input
                for no in range(N):
                    self._DeliverInput(input[0,:],no)

                # XNOR
                for no in range(N):
                    # run XNOR
                    self.PEs[no].RunXNOR()
                    count_xor += 1

                # popcount
                for no in range(N):
                    ind = t * self.P + no

                    # run popcount
                    POPCout = self.PEs[no].RunPopcount()
                    count_pop += 1

                    # pass the result to BRAM
                    self.output[0,ind] = POPCout

        # mapping uses multi PE
        elif self.mapping == 'Multi_PE':
            remaider =  self.L % self.P

            for t in range(self.T):

                N = self.P
                if t == self.T-1:
                    if remaider != 0:
                        N = remaider
                if self.L < self.P:
                    N = self.L
                TPE.append(N*self.Npe)

                # Weights
                no = 0
                for p in range(N):
                    for npe in range(self.Npe):
                        # load weights to buffer
                        ind = t*self.P*self.Npe + p*self.Npe + npe
                        row_weight = weights[ind,:]

                        # set weights to PE
                        self._DeliverWeights(row_weight,no)
                        no += 1


                # Bias
                no = 0
                for p in range(N):
                    ind = t*self.P + p
                    for npe in range(self.Npe):
                        if npe == 0:
                            self.PEs[no].SetPopArray(bias[ind],'bias')
                        else:
                            self.PEs[no].SetPopArray(0,'bias')
                        no += 1


                # Input
                for ind in range(self.Npe):
                    row_input = input[ind,:]

                    # set input to PE
                    for p in range(N):
                        no = p*self.Npe + ind
                        self._DeliverInput(row_input,no)

                # XNOR
                for no in range(N*self.Npe):
                    self.PEs[no].RunXNOR()
                    count_xor += 1


                # Popcount
                for p in range(N):
                    psum = 0
                    ind = t*self.P + p
                    for npe in range(self.Npe):
                        no = p*self.Npe + npe
                        POPCout = self.PEs[no].RunPopcount()
                        count_pop += 1
                        psum += POPCout
                    self.output[0,ind] = psum

        # Show used PE number
        f = open(self.logfile_name, 'a')
        txt = 'Mapping: {} (1 PE for one convolution)\n'.format(self.mapping)
        if self.mapping == 'Multi_PE':
            txt = 'Mapping: {} ({} PEs for one convolution)\n'.format(self.mapping, self.Npe)
        txt += 'PE Utilization Rate:  {0:9.2f} %\n'.format(100*sum(TPE)/(config.PENUMS*self.T*np.ceil(self.Npe/config.PENUMS)))
        no = 1
        txt += 'Computation Round\n'
        for numpe in TPE:
            for c in range(int(np.ceil(self.Npe/config.PENUMS))):
                if numpe > config.PENUMS:
                    txt += '    {0:<5}:{1:>4}/{2:<4}PEs\n'.format(no,config.PENUMS,config.PENUMS)
                    self.PE_util += config.PENUMS
                    self.PE_total += config.PENUMS
                    numpe = numpe - config.PENUMS
                else:
                    txt += '    {0:<5}:{1:>4}/{2:<4}PEs\n'.format(no,numpe,config.PENUMS)
                    self.PE_util += numpe
                    self.PE_total += config.PENUMS
                no += 1

        print(txt)
        f.write(txt)
        f.close()

        total_cycle, PE_cycle = self.cc.get_cycle_count(self.layer_type, w_height, i_height, 0, self.T, int(np.ceil(self.Npe/config.PENUMS)), c_flag)
        self.energy.get_energy(self.layer_type, w_height, i_height, 0, self.T, self.L, int(np.ceil(self.Npe/config.PENUMS)), c_flag, total_cycle, PE_cycle)

        # get classified results
        f = open(self.logfile_name, 'a')
        txt = '*** OUTPUT INFO ***\n'
        txt +=' Shape                     : {}\n'.format(self.output.shape)
        txt +='---------------------------------------------\n\n'
        print(txt)
        f.write(txt)
        f.close()


        if c_flag == True:
            self.output = np.argmax(self.output)
            f = open(self.logfile_name, 'a')
            txt = '\n\n\n'
            txt += '++++++++++++++++++++++++++\n'
            txt += '+  Classified Result: '+str(self.output)+'  +\n'
            txt += '++++++++++++++++++++++++++\n'
            txt += '++++++++++++++++++++++++++\n'
            txt += '+  PE Utilization: '+str(round(100*self.PE_util/self.PE_total,2))+'% +\n'
            txt += '++++++++++++++++++++++++++\n'
            print(txt)
            f.write(txt)
            f.close()

        return self.output


    ##################################################################################################
    # API
    #    Max Pooling Layer
    #    name: MaxPooling
    #    parameters:
    #        1. input
    #        2. kernel_size
    #        3. stride
    #        4. out_size
    ##################################################################################################
    def MaxPooling(self, input, kernel_size, stride, out_size):
        count = 0
        """
            Max Pooling with stride 2 and no padding

            *parameters*
            input: (1,256,h,w) numpy
            ksize: 2
        """
        _,C,H,_ = input.shape


        self.layer_type = 'MaxPooling'
        self.l = out_size
        self.ksize = kernel_size
        self.stride = stride
        self.T = np.ceil(C/self.OR_numbers) * self.l * self.l


        # output
        output = np.zeros((1, C, self.l, self.l),dtype=int)

        # Compute Max Pooling
        for i in range(self.l):
            h1 = self.stride*i
            h2 = h1 + self.ksize
            for j in range(self.l):
                w1 = self.stride*j
                w2 = w1 + self.ksize
                for c in range(C):
                    counter = np.sum(input[0,c,h1:h2,w1:w2])
                    counter = np.where(counter>0, 1, 0)
                    output[0,c,i,j] = counter
                    count += 1


        total_cycle, PE_cycle = self.cc.get_cycle_count(self.layer_type, 0, H, C, self.T, 1, False)
        self.energy.get_energy(self.layer_type, self.l, H, C, self.T, 0, 1, False, 0, 0)

        a,b,c,d = output.shape
        f = open(self.logfile_name, 'a')
        txt = '*** OUTPUT INFO ***\n'
        txt +=' Shape                     : {}\n'.format(output.shape)
        txt +='---------------------------------------------\n\n'
        print(txt)
        f.write(txt)
        f.close()

        return output

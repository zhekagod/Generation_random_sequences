import matplotlib.pyplot as plt
import numpy as np
from PhysChannel import PhysChannel, add_delays
import sys
from os import path
import importlib
from PyQt5 import QtCore, QtGui, QtWidgets

sys.path.append(path.abspath('../TimeDelay-master'))
from SigSource import SigSource
from gui import Ui_MainWindow, app, MainWindow

import ast


# TimeDelayMaster = importlib.import_module('TimeDelay-master')


def remove_from_key(d, key):
    r = dict(d)
    del r[key]
    return r


class RndInputData:
    def __init__(self):
        self.PC = None
        self.signal = None
        self.ts = None
        self.us = None
        self.with_delays = False
        self.delays = None
        self.noisy = False
        self.sig_del_noise_x = None
        self.sig_del_noise_y = None
        self.dt = 5
        # self.set_initial_data_params()

    def set_rnd_pc(self):
        self.PC = PhysChannel()

    def set_rnd_signal(self, size=100):
        self.signal = np.random.normal(0, 1, size=size)
        self.PC.set_signal(self.signal)

    def set_ts(self):
        self.ts = self.PC.gen_delays()

    def set_us(self):
        self.us = self.PC.gen_amps(self.ts)

    def set_with_delays(self, is_with_delays):
        self.with_delays = is_with_delays

    def set_old_delays(self, is_with_delays):
        self.with_delays = is_with_delays
        self.delays = add_delays(self.signal,
                                 self.ts, self.us, 5)
        self.sig_del_noise_x = self.delays[0]
        self.sig_del_noise_y = self.delays[1]

    def set_noisy(self, is_noisy, noise_coef=5):
        self.noisy = is_noisy
        self.PC.noise = noise_coef
        self.PC.noise_u = noise_coef

    def get_data_params(self):
        PC_data = self.PC.__dict__
        self_data = remove_from_key(self.__dict__, 'PC')
        return PC_data | self_data

    def gen_del_noise_sig(self):
        # self.sig_del_noise_x, self.sig_del_noise_y = np.copy(self.sig_x), np.copy(self.sig_y)
        self.sig_del_noise_x, self.sig_del_noise_y = np.copy(self.ts), np.copy(self.us)
        if self.with_delays:
            self.sig_del_noise_x, self.sig_del_noise_y = add_delays(
                self.signal, self.ts, self.us, self.dt)
        if self.noisy:
            noise = self.PC.gen_noise(len(self.sig_del_noise_y))
            self.sig_del_noise_y += noise

    def add_noise(self):
        if self.noisy:
            noise = self.PC.gen_noise(len(self.sig_del_noise_y))
            print(f'{noise=}')
            # self.sig_del_noise_x += noise
            self.sig_del_noise_y += noise

    def set_initial_data_params(self,
                                size=100,
                                is_with_delays=True,
                                is_noisy=False):
        self.set_rnd_pc()
        self.set_rnd_signal(size=size)
        self.set_ts()
        self.set_us()
        self.set_with_delays(is_with_delays)
        self.set_old_delays(is_with_delays)
        self.set_noisy(is_noisy)
        # self.add_noise()
        self.gen_del_noise_sig()
        # self.set_delays(is_with_delays)
        with open('checker.txt', 'w') as f:
            f.write('1')

    def set_data_params(self, pc_data, signal, ts, us, delays, from_file=False):
        if from_file:
            tmp = []
            for key in pc_data.keys():
                if key not in ['signal', 'ts', 'us', ]:
                    ...
        try:
            with open('checker.txt', 'r') as f:
                checker = f.read()
            if checker == '1':
                self.PC = PhysChannel()
                self.PC.set_all_params(pc_data)
                self.signal = signal
                self.ts = ts
                self.us = us
                self.delays = delays
        except FileNotFoundError:
            return


class SignalReSource(SigSource, RndInputData):
    def __init__(self):
        super().__init__()
        self.sig_mode = 'Gold'
        self.u = 1
        self.periods = 31
        self.freq = 1600
        # self.polys = ast.literal_eval('[5,3],[5,4,3,2]')
        self.gen_sig()

    def gen_sig(self):
        self.sig_x, self.sig_y, self.dt = self.generate_sig()

    def plot_del_noise_sig(self):
        sig2plot_x, sig2plot_y = self.sig_del_noise_x, self.sig_del_noise_y
        self.plot(sig2plot_x, sig2plot_y)

    def plot_sig(self):
        sig_x, sig_y, dt = self.sig_x, self.sig_y, self.dt
        sig2plot_x = sig_x
        sig2plot_y = sig_y
        self.plot(sig2plot_x, sig2plot_y)


class ExtractSignal(Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.u = 1
        self.periods = 31
        self.freq = 1600

    def signal_extractioon(self):
        self.gen_sig()
        self.gen_delays()
        self.gen_del_noise_sig()

        # self.gen_conv() # Response of correlation receiver
        # self.snr_calc()
        #self.gen_peaks()
        # self.gen_borders()
        # self.gen_spec()


def save_data_to_file(data, filename='input_signal_data.txt'):
    with open(filename, 'w') as f:
        for key in data.keys():
            f.write(f'{key}={data[key]}\n')


def parse_data_from_file(filename='input_signal_data.txt'):
    res = dict()
    flag = False
    with open(filename, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines) - 1:
            if lines[i][0].isalpha():
                partition = lines[i].split('=')
                i += 1
                try:
                    while not lines[i][0].isalpha():
                        partition[1] += lines[i]
                        i += 1
                    if tmp_clear_line[0] == '[':
                        partition[1] = tmp_clear_line + partition[1]
                except:
                    res[partition[0]] = partition[1]
                    flag = True
                if not flag:
                    tmp = ''
                    for sub in partition[1]:
                        tmp += sub.replace("\n", "")
                    partition[1] = tmp
                    tmp = partition[1].split()
                    partition[1] = tmp
                    partition[1] = partition[1][1:]
                    res[partition[0]] = partition[1]
                flag = False
    return res


if __name__ == '__main__':
    RID = RndInputData()
    is_noisy = False
    is_with_delays = True

    RID.set_initial_data_params()

    RID_params = RID.get_data_params()
    print(RID_params.keys())

    print()

    # save_data_to_file(RID_params)
    # RID_params = parse_data_from_file()
    print(RID_params)
    '''for key in RID_params.keys():
        print(key, '=', RID_params[key])'''
    '''for num in RID_params['sig']:
        print(num)
    print(RID_params['sig'])'''

    '''
    SRS_params = SignalReSource()
    SRS_params.set_initial_data_params()
    SRS_sig = SRS_params.generate_sig()
    sig_X, sig_Y, step = SRS_sig
    print(SRS_sig)
    print(sig_X, sig_Y, step)
    #SRS_params.plot_sig()
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))

    axes[0].plot(SRS_params.sig_x, SRS_params.sig_y)
    axes[1].plot(SRS_params.sig_del_noise_x, SRS_params.sig_del_noise_y)


    plt.show()
    '''
    '''
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    '''

    ui = Ui_MainWindow()
    # MainWindow.show()

    ref_signal = ExtractSignal()
    ref_signal.signal_extractioon()

    # Создаем сетку для двух графиков
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))

    sig_x, sig_y = ref_signal.sig_x, ref_signal.sig_y
    noise_sig_x, noise_sig_y = ref_signal.sig_del_noise_x, ref_signal.sig_del_noise_y
    print(f'{sig_x=}')
    print(f'{sig_y=}')

    # Построение первого графика (с шумом)
    axes[0].plot(noise_sig_x, noise_sig_y, linewidth=1.5)
    axes[0].set_xlim(-0.01, 0.42034375)
    axes[0].set_ylim(-20.1, 20.1)
    axes[0].grid()

    # Построение второго графика (без шума)
    axes[1].plot(sig_x, sig_y, linewidth=1.5)
    axes[1].set_xlim(-0.00096875, 0.02034375)
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].grid()

    plt.tight_layout()  # Для улучшения компоновки графиков
    plt.show()

    exit(0)


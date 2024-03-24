import numpy as np
from PhysChannel import PhysChannel, add_delays


class RndInputData:
    def __init__(self):
        self.PC = PhysChannel()
        self.signal = None
        self.ts = None
        self.us = None
        self.delays = None

    def set_rnd_signal(self, size=100):
        self.signal = np.random.normal(0, 1, size=size)
        PC.set_signal(self.signal)

    def set_ts(self):
        self.ts = PC.gen_delays()

    def set_us(self):
        self.us = PC.gen_amps(self.ts)

    def set_delays(self):
        self.delays = add_delays(signal, ts, us, 5)

    def set_data_params(self):
        self.set_rnd_signal()
        self.set_ts()
        self.set_us()
        self.set_delays()


def save_data_to_file(data, filename):
    with open(filename, 'w') as f:
        f.write()

PC = PhysChannel()

# Пример сигнала, который нужно установить
signal = np.random.normal(0, 1, size=100)

# Устанавливаем сигнал
PC.set_signal(signal)

ts = PC.gen_delays()
us = PC.gen_amps(ts)
delays = add_delays(signal, ts, us, 5)

print(delays)
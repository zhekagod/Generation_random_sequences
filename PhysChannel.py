import numpy as np
from scipy.stats import expon, norm, lognorm


def add_delays(sig, ts, us, dt):
    max_shift = max(ts)
    zer = np.zeros(round(max_shift/dt) + 1)
    signal = sig.tolist()
    signal.extend(zer)
    del_sig_y = np.zeros(len(signal))
    for t in range(len(ts)):
        del_sig_y[int(ts[t]/dt):int(len(sig) + int(ts[t]/dt))] += sig * us[t]
    del_sig_x = np.linspace(start=0, stop=len(del_sig_y)*dt, num=len(del_sig_y), dtype=np.float64)
    return del_sig_x, np.array(del_sig_y, dtype=np.float64)


class PhysChannel:
    def __init__(self):
        self.sig = None
        self.sig_step = None
        self.noise = 0
        self.noise_u = 0
        self.diff = 4
        self.scale = 1
        self.r0 = 10
        self.a0 = 10
        self.sigma = 9
        self.n = 2.7
        self.dist = 1000
        self.uniform = False
        self.dr = False
        self.rand = True

        # change from window
        self.diff = 5
        self.dist = 200
        self.noise_u = 5
        self.a0 = 100
        self.r0 = 100
        self.sigma = 4.5
        self.scale = 0.2

    def set_all_params(self, params):
        self.sig = params[0]
        self.sig_step = params[1]
        self.noise = params[2]
        self.noise_u = params[3]
        self.diff = params[4]
        self.scale = params[5]
        self.r0 = params[6]
        self.a0 = params[7]
        self.sigma = params[8]
        self.n = params[9]
        self.dist = params[10]
        self.uniform = params[11]
        self.dr = params[12]
        self.rand = params[13]

    def set_signal(self, signal):
        self.sig = signal

    def gen_noise(self, size):
        self.noise = np.random.normal(0, self.noise_u, size=size)
        return self.noise

    def gen_delays(self):
        if self.uniform is False:
            ts = expon.rvs(size=self.diff, scale=self.scale)
            ts = np.sort(ts)
            if len(ts):
                for i in range(len(ts)):
                    ts[-(i + 1)] = np.sum(ts[0:len(ts) - i])
        else:
            ts = np.arange(start=self.scale, stop=self.scale * self.diff + self.scale / 10, step=self.scale)
        if self.dr:
            d_r = np.array([0])
            ts = np.append(d_r, ts)
        return ts

    def gen_amps(self, ts):
        r = self.r0 + self.dist
        us = []
        start = 0
        if self.dr:
            start = 1
        for y in range(len(ts)):
            if y == 0:
                r += 300 * ts[y]
            else:
                r += 300 * (ts[y] - ts[y - 1])
            u = self.a0 * (self.r0 / r) ** self.n
            us.append(u)
        us = np.array(us)

        if self.rand:
            for u in range(start, len(us)):
                y = np.random.normal(0, self.sigma, size=1)
                alpha = 10 ** (y / 20)
                us[u] *= alpha
        return us



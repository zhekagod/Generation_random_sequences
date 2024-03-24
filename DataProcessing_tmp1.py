import numpy as np
import matplotlib.pyplot as plt
import math as m
from PhysChannel import PhysChannel, add_delays

PC = PhysChannel()

# Пример сигнала, который нужно установить
signal = np.random.normal(0, 1, size=100)

# Устанавливаем сигнал
PC.set_signal(signal)

ts = PC.gen_delays()
us = PC.gen_amps(ts)
delays = add_delays(signal, ts, us, 5)

print(delays)


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Signal', color=color)
ax1.plot(delays[0], delays[1], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Amplitude', color=color)
ax2.plot(delays[0], delays[1], '.', color=color)  # Увеличиваем точность отображения, чтобы график был более детализированным
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()



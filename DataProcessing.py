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


# Данные задержек и амплитуд
delays_x, delays_y = delays

# Построение графика задержек
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(delays_x, delays_y, color='blue')
plt.title('График задержек')
plt.xlabel('Время (с)')
plt.ylabel('Задержка (с)')

# Данные амплитуд
amps_x = np.arange(len(us)) * 5  # Шаг между амплитудами равен 5 (как в add_delays)

# Построение графика амплитуд
plt.subplot(2, 1, 2)
plt.bar(amps_x, us)
# plt.stem(amps_x, us, linefmt='blue', markerfmt='bo', basefmt=' ')
plt.title('График амплитуд')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')

plt.tight_layout()
plt.show()



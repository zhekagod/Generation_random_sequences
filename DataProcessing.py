import numpy as np
import matplotlib.pyplot as plt
import math as m
from GenData import RndInputData, save_data_to_file, parse_data_from_file

'''
PC = PhysChannel()

# Пример сигнала, который нужно установить
signal = np.random.normal(0, 1, size=100)

# Устанавливаем сигнал
PC.set_signal(signal)

ts = PC.gen_delays()
us = PC.gen_amps(ts)
delays = add_delays(signal, ts, us, 5)

print(delays)
'''
RID = RndInputData()

RID.set_initial_data_params(is_with_delays=True,
                            is_noisy=False)

# Создание сетки для 4 графиков
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))

for noise_param, ax_row in zip([False, True], axes):

    if noise_param:
        RID.set_noisy(noise_param)
        RID.gen_del_noise_sig()
        RID.add_noise()

    # Данные задержек и амплитуд
    ts = RID.ts
    us = RID.us
    if not noise_param:
        delays_x, delays_y = RID.delays
    else:
        delays_x, delays_y = RID.sig_del_noise_x, RID.sig_del_noise_y

    # Построение графика задержек
    ax_row[0].plot(delays_x, delays_y, color='blue')
    ax_row[0].set_title('График задержек')
    ax_row[0].set_xlabel('Время (с)')
    ax_row[0].set_ylabel('Задержка (с)')

    # Данные амплитуд
    amps_x = np.arange(len(us)) * 5  # Шаг между амплитудами равен 5 (как в add_delays)

    # Построение графика амплитуд
    ax_row[1].bar(delays_x, delays_y)
    ax_row[1].set_title('График амплитуд')
    ax_row[1].set_xlabel('Время (с)')
    ax_row[1].set_ylabel('Амплитуда')

plt.tight_layout()
plt.show()



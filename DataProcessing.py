import numpy as np
import matplotlib.pyplot as plt
import math as m
from GenData import RndInputData, save_data_to_file, parse_data_from_file
from GenData import ExtractSignal


def colored_print(text, color='red'):
    text_colors = {'black': '\033[30m',
                   'red': '\033[31m',
                   'green': '\033[32m',
                   'yellow': '\033[33m',
                   'blue': '\033[34m',
                   'purple': '\033[35m',
                   'Turquoise': '\033[36m',
                   'white': '\033[37m',
                   'reset': '\033[0m'}

    if color == 'red':
        print(text_colors[color] + '{}'.format(text), end='')
        print(text_colors['reset'].format(text))


def find_extremes(x_values, y_values, extreme_type='all', first=False):
    extremes = []
    for i in range(1, len(y_values) - 1):
        if extreme_type == 'all':
            if ((y_values[i - 1] <= y_values[i]) and
                (y_values[i] >= y_values[i + 1])) or \
                    ((y_values[i - 1] >= y_values[i]) and
                     (y_values[i] <= y_values[i + 1])):
                extremes.append([x_values[i], y_values[i]])
        elif extreme_type == 'positive':
            if (y_values[i - 1] <= y_values[i]) and \
                    (y_values[i] >= y_values[i + 1]):
                extremes.append([x_values[i], y_values[i]])
        elif extreme_type == 'negative':
            if (y_values[i - 1] >= y_values[i]) and \
                    (y_values[i] <= y_values[i + 1]):
                extremes.append([x_values[i], y_values[i]])
        if first:
            break
    return extremes


def generate_sequence(bit_counts: int,
                      values: list[float] | np.array([float]),
                      delay_delta: float):
    sequence = ''
    # Написано, что ключевые биты генерируются
    # при оценки по одному каналу
    for i in range(1, bit_counts):
        if values[i] - values[i - 1] - delay_delta >= 0:
            sequence += '1'
        elif values[i] - values[i - 1] - delay_delta < 0:
            sequence += '0'
    return bin(int(sequence, 2))


def find_peak(x_values, y_values):
    for x in x_values:
        ...


def plot_convolution(signal_source,
                     instant_show=False,
                     figure=None,
                     plot_axis=None,
                     add_noise=False,
                     need_to_show_now=False):
    if instant_show:
        plt.show()
    sig_x = np.append(signal_source.sig_del_noise_x, np.array([max(signal_source.sig_del_noise_x) + signal_source.dt]))
    sig_y = np.append(np.abs(signal_source.conv), np.zeros(1))
    print(f'{sig_x=}')
    print(f'{sig_y=}')
    # self.axes[1][0].plot(sig_x, sig_y)
    if add_noise:
        # noise = self.phys_channel.gen_noise(size=len(sig_x))
        noise = signal_source.gen_sig_noise(len(sig_x), 0.001)
        noise[0] = 0  # Надо ли обнулять первую координату???
        print(f'{noise=}')
        sig_x += 0.1
        print(f'noise_x={sig_x}')
    if need_to_show_now:
        plt.plot(sig_x, sig_y)
        plt.show()
    else:
        if (figure is not None) and (plot_axis is not None):
            #  if (figure and plot_axis) is not None
            if not add_noise:
                plot_axis[0].plot(sig_x, sig_y)
            else:
                plot_axis[1].plot(sig_x, sig_y)


'''RID = RndInputData()

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
plt.show()'''

# ui = Ui_MainWindow()
# MainWindow.show()


ref_signal = ExtractSignal()
ref_signal.signal_extraction()
print(f'{ref_signal.peaks_x=}')
print(f'{ref_signal.peaks_y=}')
try:
    print(f'{ref_signal.sig_x == ref_signal.sig_del_noise_x}')
except ValueError:
    colored_print(f'ValueError with comparison sig_x and sig_del_noise_x was excepted')

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 5))

plot_convolution(ref_signal, figure=fig,
                 plot_axis=axes)
plot_convolution(ref_signal, figure=fig,
                 plot_axis=axes,
                 add_noise=True)
plot_convolution(ref_signal, instant_show=True)

# ref_signal.show_signal()

print(f'{ref_signal.sig_fft_y}')

# Создаем сетку для двух графиков
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 5))

sig_x, sig_y = ref_signal.sig_x, ref_signal.sig_y
sig_length = len(sig_x)
noise_sig_x, noise_sig_y = ref_signal.sig_del_noise_x, ref_signal.sig_del_noise_y
# print(f'{sig_x=}')
# print(f'{sig_y=}')
print(f'{sig_length=}')

# Построение первого графика (с шумом)
axes[0].plot(noise_sig_x, noise_sig_y, linewidth=1.5)
axes[0].set_xlabel("График с шумом")
axes[0].set_xlim(-0.01, 0.42034375)
axes[0].set_ylim(-20.1, 20.1)
axes[0].grid()

# Построение второго графика (без шума)
axes[1].plot(sig_x, sig_y, linewidth=1.5)
axes[1].set_xlabel("Референсный сигнал")
axes[1].set_xlim(-0.00096875, 0.02034375)
axes[1].set_ylim(-1.1, 1.1)
axes[1].grid()

plt.tight_layout()  # Для улучшения компоновки графиков

print(*sig_y[:1000])
extreme_values = find_extremes(sig_x, sig_y,
                               extreme_type='all')
print(extreme_values[:1000])
print('=' * 200)

# Инициализируем необходимые данные
bits_count = 16  # количество бордюров, битность выходных данных???
nq = bits_count - 1  # количество разделителей
sum_error = 0
output_q_intervals = []
output_q_data = []
output_q_book = []

# 1. Находим максимум и минимум во входных данных

# Находим максимальное значение y в экстремумах
max_extr = max(extreme_values, key=lambda x: x[1])[1]
max_exrt_idx = extreme_values.index([item for item in extreme_values if item[1] == max_extr][0])
print(f'{max_extr=}, {max_exrt_idx=}, {extreme_values[max_exrt_idx]=}')

# Находим минимальное значение y в экстремумах
min_extr = min(extreme_values, key=lambda x: x[1])[1]
min_extr_idx = extreme_values.index([item for item in extreme_values if item[1] == min_extr][0])
print(f'{min_extr=}, {min_extr_idx=}, {extreme_values[min_extr_idx]=}')

# 2. Рассчитываем размер окна между интервалами
delta = max_extr - min_extr
interval_size = delta / bits_count
# задержки --- расстояние между экстремумами в x-ах
delays = [second[0] - first[0] for first, second in zip(extreme_values[:-1], extreme_values[1:])]
print(f'{delta=}, {interval_size=}')
print(f'{delays[:10]=}')

# 3. Производим квантование и подсчет ошибки
for i in range(sig_length):
    output_q_intervals.append((sig_y[i] - min_extr) / interval_size)
    output_q_data.append(output_q_intervals[i] * interval_size + min_extr)
    sum_error += abs(output_q_data[i] - sig_y[i]) / delta

for i in range(nq):
    match i:
        case 0:
            output_q_book.append(min_extr + interval_size * 0.5)
        case _:
            output_q_book.append(output_q_book[i - 1] + interval_size)

# 4. Выводим ошибку
sum_error /= sig_length
sum_error *= 100
print(f'{sum_error=:.30f}')
plt.show()
exit(0)

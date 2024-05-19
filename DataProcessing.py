import numpy as np
import matplotlib.pyplot as plt
import math as m
import random as rnd
from GenData import RndInputData, save_data_to_file, parse_data_from_file
from GenData import ExtractSignal
import pyqtgraph as pg


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


def Q_Bit():
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


def mean_quantisation_old(array, length, bits_count,
                          include_remains=False,
                          where_include='center'):
    step = length // bits_count
    if include_remains and length % bits_count != 0:
        if where_include == 'center':
            # Рассчитываем количество дополнительных элементов для центрального подсписка
            add_elements = max(bits_count, bits_count - (length % bits_count) + 1)

            # Индекс центрального подсписка
            center_index = bits_count // 2

            # Разбиваем массив на подсписки до центра
            res = [array[step * i:step * (i + 1)] for i in range(center_index)]

            # Добавляем дополнительные элементы в центральный подсписок
            center_sublist = array[step * center_index:step * (center_index + 1) + add_elements]
            res.append(center_sublist)

            # Продолжаем разбиение на подсписки после центрального
            res += [array[step * (i + 1) + add_elements:step * (i + 2) + \
                                                        add_elements] for i in range(center_index, bits_count + 2)]

            return res
        elif where_include == 'left':
            # Рассчитываем количество дополнительных элементов для левого подсписка
            add_elements = bits_count - (length % bits_count) + 1

            # Разбиваем массив на подсписки
            res = [array[add_elements + step * i:step * (i + 1) + add_elements] for i in range(1, bits_count)]

            # Добавляем дополнительные элементы в левый подсписок
            left_sublist = array[:step + add_elements]
            res.insert(0, left_sublist)

            return res
        elif where_include == 'right':
            # Рассчитываем количество дополнительных элементов для правого подсписка
            add_elements = bits_count - (length % bits_count) + 1

            # Разбиваем массив на подсписки
            res = [array[step * i:step * (i + 1)] for i in range(bits_count - 1)]

            # Добавляем дополнительные элементы в правый подсписок
            right_sublist = array[-add_elements - step:]
            res.append(right_sublist)

            return res
    else:
        return [array[step * i:step * (i + 1)] for i in range(bits_count)]

    '''
    res = []
    l, n = length, bits_count
    step = 0
    for i in range(n):
        res.append(array[step:])
    '''


def mean_quantisation(array, length, bits_count,
                      include_remains=False,
                      where_include='center'):
    step = length // bits_count
    extended_size = step
    if include_remains:
        # Размер каждого подмассива, кроме выбранного для расширения
        if where_include == 'left' or where_include == 'right':
            size = length // (bits_count - 1)
        else:
            size = length // bits_count

        # Размер выбранного для расширения подмассива
        if where_include == 'left':
            extended_size = size + length % (bits_count - 1)
            bits_count -= 1
        elif where_include == 'center':
            extended_size = size + length % bits_count
        elif where_include == 'right':
            extended_size = size + length % (bits_count - 1)

        # Разделение на подмассивы
        result = []
        start = 0
        for i in range(bits_count):
            # Размер текущего подмассива
            if where_include == 'left' and i == 0:
                end = start + extended_size
            elif where_include == 'center' and i == bits_count // 2:
                end = start + extended_size
            elif where_include == 'right' and i == bits_count - 1:
                end = start + extended_size
            else:
                # Обычные подмассивы
                end = start + size
            # Добавление подмассива в результат
            result.append(array[start:end])
            # Обновление начального индекса для следующего подмассива
            start = end

        return result
    else:
        return [array[step * i:step * (i + 1)] for i in range(bits_count)]


'''# Пример использования
array = list(range(1, 41))
bits_count = 12
result = split_array(array, bits_count, where_include='right')
print(result)'''


def rnd_quantisation(array, total_length, bits_count, min_length=1, max_length=None):
    max_length = max_length or total_length  # Если не указана максимальная длина, используем всю длину исходного массива
    if min_length > max_length:
        raise ValueError("Min length cannot be greater than max length.")

    # Генерация случайных длин подмассивов
    lengths = [rnd.randint(min_length, min(max_length, total_length)) for _ in range(bits_count)]
    # Корректировка последней длины, чтобы сумма равнялась длине массива
    lengths[-1] += len(array) - sum(lengths)

    # Разделение на подмассивы
    result = []
    start = 0
    for length in lengths:
        end = start + length
        result.append(array[start:end])
        start = end

    return result


def generate_numbers(bits_count, length):
    numbers = []

    # Генерация случайных чисел
    for _ in range(bits_count - 1):
        num = rnd.randint(1, length - (bits_count - len(numbers)))
        numbers.append(num)
        length -= num

    # Последнее число - остаток, чтобы обеспечить сумму равную length
    numbers.append(length)

    return numbers


'''
x1, x2 = generate_numbers(8, 41), None
print(f'{x1=}, {x2=}')
'''


def quantisation(array, bits_count, q_type='mean', **kwargs):
    # mean означает equal intervals
    # quant_types = {}
    # print(kwargs)
    if q_type == 'mean':
        q = mean_quantisation(array, len(array),
                              bits_count, **kwargs)
        # print(f'{q=}')
        # print(f'{len(q)=}')
        # print(f'{len(array)=}')
    elif q_type == 'numpy_array_split':
        q = np.array_split(array, len(array) // bits_count)
        # print(f'{q=}')
    elif q_type == 'rnd':
        q = rnd_quantisation(array, len(array), bits_count)
        # print(f'rnd {q=}')

    return q


def generate_sequence(bits_count: int,
                      values: list[float] | np.array([float]) = None,
                      delay_delta: float = None,
                      book=False,
                      **kwargs):
    if book:
        sequence = ''
        # Написано, что ключевые биты генерируются
        # при оценки по одному каналу
        for i in range(1, bits_count):
            if values[i] - values[i - 1] - delay_delta >= 0:
                sequence += '1'
            elif values[i] - values[i - 1] - delay_delta < 0:
                sequence += '0'
        return bin(int(sequence, 2))
    else:
        res = [bin(num) for num in range(bits_count)]
        rnd.shuffle(res)
        return res


def generate_random_colors(count, seed=1):
    rnd.seed(seed)
    colors = []
    for _ in range(count):
        color = "#{:06x}".format(rnd.randint(0, 0xFFFFFF))
        colors.append(color)
    return colors


def find_interval_distance(array: list[list[float]] | np.array((np.array([float]))),
                           first: int, second: int,
                           idx: int = 0):
    try:
        return abs(array[second][idx] - array[first][idx])
    except IndexError:
        print(first, second)


def mean_interval_distance(array: list[float] | np.array([float]), length: int):
    res = 0
    for i in range(length - 1):
        res += find_interval_distance(array, first=i, second=i + 1,
                                      idx=len(array[0]) // 2)
    return res / length


def find_peak(x_values, y_values):
    for x in x_values:
        ...


def plot_sequence_borders(seq, plot_axis, axis_num):
    '''# Определяем длину интервала и рисуем его на графике
    interval_lengths = [subseq[-1] - subseq[0] for subseq in seq]
    for i, length in enumerate(interval_lengths):
        # axhline, axvline
        plot_axis[axis_num].axhline(0.1 * i, xmin=0, xmax=length*100, color='red', linewidth=2)'''
    for interval in seq:
        # plot_axis[axis_num].text(interval[0], 10e-4, '---'*len(interval), fontsize=3, ha='center')
        plot_axis[axis_num].plot(interval[0], linestyle='--')
        plot_axis[axis_num].axhline(0.001,
                                    xmin=interval[0],
                                    xmax=interval[-1],
                                    linewidth=10)
def plot_peaks_bar(signal_source,
                   add_noise=False,
                   plot_axis=None,
                   axis_num=None,
                   need_to_show_now=False,
                   **kwargs):
    '''signal_source.preview_plt = pg.PlotWidget()
    sig_s_prev = signal_source.preview_plt
    sig_s_prev.setBackground('w')
    sig_s_prev.setFixedSize(180, 145)
    sig_s_prev.move(20, 490)
    sig_s_prev.showGrid(x=True, y=True)'''
    sig_orig_x = signal_source.opt_rec.orig_array_x
    pos_x = kwargs['len_bar']
    if ('bar_colors' in kwargs) and len(kwargs['bar_colors']) > 0:
        bar_colors = kwargs['bar_colors']
    else:
        bar_colors = generate_random_colors(len(pos_x),
                                            seed=kwargs['bar_colors_seed'])
    if add_noise:
        if 'noise' in kwargs:
            pos_x += kwargs['noise']
        else:
            if 'noise_limits' in kwargs:
                noise = [rnd.uniform(
                    kwargs['noise_limits'][0],
                    kwargs['noise_limits'][1])
                    for _ in range(len(pos_x))]
                for i, pos in enumerate(pos_x):
                    if (pos + noise[i] < sig_orig_x[0]) and (noise[i] < 0):
                        pos_x[i] = sig_orig_x[-1] + noise[i]
                    elif (pos + noise[i] >= sig_orig_x[-1]) and (noise[i] > 0):
                        pos_x[i] = sig_orig_x[0] + noise[i]
                    else:
                        pos_x[i] += noise[i]

    if plot_axis is None or axis_num is None:
        plt.grid(True)
        plt.bar(
            x=pos_x,
            height=kwargs['height_bar'],
            width=kwargs['width_bar'],
            color=bar_colors)
        if need_to_show_now:
            plt.show()
    else:
        plot_axis[axis_num].grid(True)
        plot_axis[axis_num].set_xlabel(kwargs['x_axis_label'])
        plot_axis[axis_num].set_ylabel(kwargs['y_axis_label'])
        plot_axis[axis_num].bar(
            x=pos_x,
            height=kwargs['height_bar'],
            width=kwargs['width_bar'],
            color=bar_colors)
        if need_to_show_now:
            plot_axis[axis_num].show()


def plot_convolution(signal_source,
                     instant_show=False,
                     figure=None,
                     with_bars=False,
                     plot_axis=None,
                     add_noise=False,
                     need_to_show_now=False):
    if instant_show:
        plt.show()
        return
    sig_x = np.copy(signal_source.opt_rec.orig_array_x)
    sig_y = np.copy(signal_source.opt_rec.orig_array_y)
    print(f'{sig_x=}')
    print(f'{sig_y=}')
    # self.axes[1][0].plot(sig_x, sig_y)
    # self.peaks_x, self.border_x, self.border_y, self.height = self.opt_rec.find_peaks_()
    extreme_data = signal_source.opt_rec.find_peaks_()
    extreme_values = extreme_data[0]
    print(f'{extreme_values=}')
    colored_print(f'{len(sorted(sig_x[extreme_values])) == len(signal_source.peaks_x)}')
    if add_noise:
        # noise = self.phys_channel.gen_noise(size=len(sig_x))
        noise = signal_source.gen_sig_noise(len(sig_x), 0.001)
        noise[0] = 0  # Надо ли обнулять первую координату???
        print(f'{noise=}')
        # 0.1
        sig_x += noise
        # signal_source.peaks_x += noise
        print(f'noise_x={sig_x}')
    if need_to_show_now:
        plt.plot(sig_x, sig_y)
        plt.show()
    else:
        if (figure is not None) and (plot_axis is not None):
            if not with_bars:
                #  if (figure or plot_axis) is not None
                if not add_noise:
                    # plot_axis[0].autoscale(False, axis='x')
                    plot_axis[0].plot(sig_x, sig_y)
                else:
                    # plot_axis[1].autoscale(False, axis='x')
                    print(signal_source.opt_rec.orig_array_x[0], sig_x[-1])
                    plot_axis[1].set_xlim(signal_source.opt_rec.orig_array_x[0], sig_x[-1])
                    plot_axis[1].plot(sig_x, sig_y)
            else:
                if not add_noise:
                    plot_axis[0].plot(sig_x, sig_y)
                    plot_peaks_bar(
                        signal_source,
                        add_noise=False,
                        plot_axis=plot_axis, axis_num=1,
                        len_bar=signal_source.ts,
                        height_bar=signal_source.us,
                        width_bar=len(signal_source.sig_x) * (signal_source.sig_x[1] - signal_source.sig_x[0]),
                        bar_colors='red')

                else:
                    # plot_axis[1].autoscale(False, axis='x')
                    # print(signal_source.opt_rec.orig_array_x[0], sig_x[-1])
                    plot_axis[2].set_xlim(signal_source.opt_rec.orig_array_x[0], sig_x[-1])
                    plot_axis[2].plot(sig_x, sig_y)
                    plot_peaks_bar(
                        signal_source,
                        add_noise=False,
                        plot_axis=plot_axis, axis_num=2,
                        len_bar=ts,
                        height_bar=us,
                        width_bar=len(signal_source.sig_x) * (signal_source.sig_x[1] - signal_source.sig_x[0]),
                        bar_colors='red')


def determine_peak_sequences(peaks_x, sequences):
    idx_of_seq = []
    for peak_x in peaks_x:
        for idx, seq in enumerate(sequences):
            if peak_x in seq:
                print(f'{peak_x=}, {seq=}')
                idx_of_seq.append(idx)
    return [sequences[i] for i in idx_of_seq]


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

if __name__ == '__main__':
    ref_signal = ExtractSignal()
    ref_signal.signal_extraction()
    print(f'{ref_signal.peaks_x=}, {ref_signal.ts=}')
    print(f'{ref_signal.peaks_y=}, {ref_signal.us=}')
    bar_colors_seed = rnd.randint(0, 256)

    # Проведение квантизации
    opt_rec_orig_x = ref_signal.opt_rec.orig_array_x
    len_opt_rec_orig_x = len(opt_rec_orig_x)
    quants = quantisation(opt_rec_orig_x, 1024,
                          include_remains=True, where_include='center')
    print(quants[-1][-1] == opt_rec_orig_x[-1])

    print(generate_sequence(1024))

    m_intv_distance = mean_interval_distance(quants, 1024)
    print(f'{m_intv_distance=}')

    peak_sequences = determine_peak_sequences(ref_signal.peaks_x, quants)
    print(f'{peak_sequences=}')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    plot_sequence_borders(peak_sequences, axes, 0)

    # цвета должны соответствовать столбца в независимости от сдвигов
    # и прыжков (отфиксировать строемую длину???)
    # зафиксировать оси для обоих bar plot-ов (максимальый масштаб)
    # передавать пики или ts и us ???
    plot_peaks_bar(ref_signal,
                   plot_axis=axes,
                   axis_num=0,
                   need_to_show_now=False,
                   len_bar=ref_signal.peaks_x,
                   height_bar=ref_signal.peaks_y,
                   width_bar=len(ref_signal.sig_x) * (ref_signal.sig_x[1] - ref_signal.sig_x[0]),
                   x_axis_label='Исходное расположение пиков',
                   y_axis_label='Высота пиков',
                   bar_colors_seed=bar_colors_seed
                   )

    plot_peaks_bar(ref_signal,
                   add_noise=True,
                   plot_axis=axes,
                   axis_num=1,
                   need_to_show_now=False,
                   len_bar=ref_signal.peaks_x,
                   height_bar=ref_signal.peaks_y,
                   width_bar=len(ref_signal.sig_x) * (ref_signal.sig_x[1] - ref_signal.sig_x[0]),
                   noise_limits=[-0.1, 0.1],
                   x_axis_label='Сдвинутые шумом пики',
                   y_axis_label='Высота пиков',
                   bar_colors_seed=bar_colors_seed
                   )

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

    '''
    001252470560381568
    0008271170301196819
    '''
    '''
    0.0008628076139691179
    '''
    # quantisation(list(range(1, 101)), 10)
    '''
    quantisation(list(range(1, 12)), 3, include_remains=True,
                 where_include='center')
    quantisation(list(range(1, 12)), 3, include_remains=True,
                 where_include='left')
    quantisation(list(range(1, 12)), 3, include_remains=True,
                 where_include='right')
    '''

    '''# Пример использования
    array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    bits_count = 4
    result = quantisation(array, bits_count, q_type='rnd')'''

    '''
    quantisation(list(range(1, 41)), 12, include_remains=True,
                 where_include='center')
    quantisation(list(range(1, 41)), 12, include_remains=True,
                 where_include='left')
    quantisation(list(range(1, 41)), 12, include_remains=True,
                 where_include='right')
    quantisation(list(range(1, 48)), 12, include_remains=True)
    '''
    # min(bits_count - (length % bits_count),
    '''
    quantisation(ref_signal.opt_rec.orig_array_x,
                 16,
                 include_remains=True)
    '''
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

    '''
    print(*sig_y[:1000])
    extreme_values = find_extremes(sig_x, sig_y,
                                   extreme_type='all')
    print(extreme_values[:1000])
    print('=' * 200)
    '''

    plt.show()
    exit(0)

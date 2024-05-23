import numpy as np
import matplotlib.pyplot as plt
import math as m
import random as rnd
from PhysChannel import add_delays
from GenData import RndInputData, save_data_to_file, parse_data_from_file
from GenData import ExtractSignal
import pyqtgraph as pg
import copy


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
                      where_include='center',
                      **kwargs):
    if bits_count == 1:
        return [array]
    elif bits_count == 2:
        return [array[0:length//2], array[length//2::]]
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


def array_extension(array, extent):
    # Проверяем, что extent корректный
    if extent < 0 or extent > len(array) - 1:
        raise ValueError("Extent должен быть положительным и не больше len(array) - 1")

    # Преобразуем список в массив NumPy, если это необходимо
    if isinstance(array, list):
        array = np.array(array)

    # Создаем результирующий массив с достаточным размером
    extended_array = []

    # Перебираем пары соседних элементов и вставляем между ними средние значения
    for i in range(len(array) - 1):
        extended_array.append(array[i])

        # Вставляем средние значения extent раз
        for j in range(extent):
            weight = (j + 1) / (extent + 1)
            average_value = (1 - weight) * array[i] + weight * array[i + 1]
            extended_array.append(average_value)

    # Добавляем последний элемент
    extended_array.append(array[-1])

    # Преобразуем обратно в list, если исходный массив был list
    if isinstance(array, np.ndarray):
        return np.array(extended_array)
    else:
        return extended_array


def array_mean_extension(array, extent):
    # Проверяем, что extent корректный
    if extent < 0 or extent > len(array) - 1:
        raise ValueError("Extent должен быть положительным и не больше len(array) - 1")

    # Преобразуем список в массив NumPy, если это необходимо
    if isinstance(array, list):
        array = np.array(array)

    # Создаем результирующий массив с достаточным размером
    extended_array = []

    # Перебираем пары соседних элементов и вставляем между ними среднее значение extent раз
    for i in range(len(array) - 1):
        extended_array.append(array[i])

        if i < extent:
            average_value = (array[i] + array[i + 1]) / 2
            extended_array.append(average_value)

    # Добавляем последний элемент
    extended_array.append(array[-1])

    # Преобразуем обратно в list, если исходный массив был list
    if isinstance(array, np.ndarray):
        return np.array(extended_array)
    else:
        return extended_array


def min_squares(x, y, z):
    X = np.column_stack((x, y))

    # Добавляем столбец из единиц, чтобы учесть свободный член
    X = np.column_stack((X, np.ones_like(x)))

    # Находим коэффициенты уравнения плоскости с помощью метода наименьших квадратов
    coefficients = np.linalg.lstsq(X, z, rcond=None)[0]

    # Коэффициенты уравнения плоскости: z = ax + by + c
    a, b, c = coefficients
    return a, b, c

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
                      to_shuffle=True,
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
        max_length = max(len(sequence).bit_count(), bits_count.bit_count())
        sequence = sequence.zfill(int(m.log2(max_length)))
        return bin(int(sequence, 2))
    else:
        res = ['0b' + bin(num)[2:].zfill(int(m.log2(bits_count))) for num in range(bits_count)]
        if to_shuffle:
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


def gen_peaks_bar(signal_source, add_noise=False, **kwargs):
    sig_orig_x = signal_source.opt_rec.orig_array_x
    pos_x = copy.deepcopy(kwargs['len_bar'])
    noise = np.zeros(len(pos_x))
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
                '''noise = [np.random.uniform(
                    kwargs['noise_limits'][0],
                    kwargs['noise_limits'][1])
                    for _ in range(len(pos_x))]'''
                noise = np.random.uniform(
                    kwargs['noise_limits'][0],
                    kwargs['noise_limits'][1], len(pos_x))
                if 'extend_axis_x' in kwargs and kwargs['extend_axis_x']:
                    for i, pos in enumerate(pos_x):
                        if (pos + noise[i] < sig_orig_x[0]) and (noise[i] < 0):
                            pos_x[i] = sig_orig_x[-1] + noise[i]
                        elif (pos + noise[i] >= sig_orig_x[-1]) and (noise[i] > 0):
                            pos_x[i] = sig_orig_x[0] + noise[i]
                        else:
                            pos_x[i] += noise[i]
                else:
                    for i in range(len(pos_x)):
                        pos_x[i] += noise[i]
    return pos_x, bar_colors, noise


def plot_peaks_bar(pos_x, bar_colors,
                   plot_axis=None, axis_num=None,
                   bit_seqs=None,
                   vertical_bit_seqs=False,
                   **kwargs):
    if plot_axis is None or axis_num is None:
        plt.grid(True)
        bars = plt.bar(
            x=pos_x,
            height=kwargs['height_bar'],
            width=kwargs['width_bar'],
            color=bar_colors)
        if bit_seqs is not None:
            if vertical_bit_seqs:
                for bar, bit_seq in zip(bars, bit_seqs):
                    plt.text(bar.get_x() + bar.get_width(),
                             bar.get_height(),
                             bit_seq,
                             ha='left',
                             va='bottom',
                             rotation=90)
            else:
                # Добавление подписей к столбикам
                for i, bar in enumerate(min(bars, bit_seqs, key=len)):
                    bar.set_label(bit_seqs[i])  # Устанавливаем подпись для каждого столбика
                plt.legend(loc='upper right')  # Отображаем легенду
        if kwargs['need_to_show_now']:
            plt.show()
    else:
        plot_axis[axis_num].grid(True)
        plot_axis[axis_num].set_xlabel(kwargs['x_axis_label'])
        plot_axis[axis_num].set_ylabel(kwargs['y_axis_label'])
        bars = plot_axis[axis_num].bar(
            x=pos_x,
            height=kwargs['height_bar'],
            width=kwargs['width_bar'],
            color=bar_colors)
        if bit_seqs is not None:
            if vertical_bit_seqs:
                for bar, bit_seq in zip(bars, bit_seqs):
                    plot_axis[axis_num].text(bar.get_x(),
                                             bar.get_height(),
                                             bit_seq,
                                             ha='left',
                                             va='bottom',
                                             rotation=90)
            else:
                try:
                    for i in range(len(min(bars, bit_seqs, key=len))):
                        # AttributeError: 'str' object has no attribute 'set_label'
                        bars[i].set_label(bit_seqs[i])  # Устанавливаем подпись для каждого столбика
                except AttributeError:
                    ...
                plot_axis[axis_num].legend(loc='upper right')  # Отображаем легенду
        if kwargs['need_to_show_now']:
            plt.show()


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


def determine_peak_sequences(peaks_x, sequences, bits):
    idx_of_seq = []
    if len(sequences) == 1 and len(bits) == 1:
        idx_of_seq.append(bits[0])
        return idx_of_seq
    try:
        for peak_x in peaks_x:
            for idx, seq in enumerate(sequences):
                if seq[0] <= peak_x <= seq[-1]:
                    idx_of_seq.append(idx)
        # print(f'{idx_of_seq=}')
        return [bits[i] for i in idx_of_seq]
    except IndexError:
        return idx_of_seq


def counting_equal_bits(bit_num1, bit_num2):
    equal_count, non_equal_count = 0, 0
    if len(bit_num1) == len(bit_num2) and len(bit_num2) == 3 and \
        bit_num1 == bit_num2:
        return 3, 0
    for bit in zip(list(bit_num1[2:]), list(bit_num2[2:])):
        if bit[0] == bit[1]:
            equal_count += 1
        else:
            non_equal_count += 1
    return equal_count, non_equal_count


# Поменять на нормальные названия везде (comparing или comparison???)
def comparing_bit_sequences(sequences1, sequences2,
                            comparison_type='percent',
                            **kwargs):
    try:
        if comparison_type == 'percent':
            percents = []
            for seq_group in zip(sequences1, sequences2):
                # Будет ли Missing Argument??? Нет такого =D
                percent = counting_equal_bits(*seq_group)[0] / len(seq_group[0]) * 100
                percents.append(round(percent, 3))
            return percents, sum(percents) / len(percents)
        elif comparison_type == 'full_equal':
            equals = []
            for seq_group in zip(sequences1, sequences2):
                # Будет ли Missing Argument??? Нет такого =D
                if counting_equal_bits(*seq_group)[0] / len(seq_group[0]) * 100 >= 80:
                    equals.append(1)
                else:
                    equals.append(0)
            return equals, sum(equals) / len(equals)
    except ZeroDivisionError:
        return [], 0


# Код Хеминга?
def correct_bits(sequence):
    ...


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
    count_peaks = 15
    ref_signal.phys_channel.diff = count_peaks
    ref_signal.signal_extraction()
    print(ref_signal.phys_channel.diff)
    opt_rec_orig_x = ref_signal.opt_rec.orig_array_x
    len_opt_rec_orig_x = len(opt_rec_orig_x)
    print(f'{ref_signal.peaks_x=}, {ref_signal.ts=}')
    print(f'{ref_signal.peaks_y=}, {ref_signal.us=}')
    bar_colors_seed = rnd.randint(0, 256)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # plot_sequence_borders(peak_sequences, axes, 0)

    # цвета должны соответствовать столбца в независимости от сдвигов
    # и прыжков (отфиксировать строемую длину???)
    # зафиксировать оси для обоих bar plot-ов (максимальый масштаб)
    # передавать пики или ts и us ???
    (orig_bar_peaks_x,
     orig_bar_colors, _) = gen_peaks_bar(
        ref_signal,
        len_bar=ref_signal.ts,
        bar_colors_seed=bar_colors_seed)
    (noise_bar_peaks_x,
     noise_bar_colors, noise_bar_x) = gen_peaks_bar(
        ref_signal,
        add_noise=True,
        # 0.0009866208565240113
        noise_limits=[-0.1, 0.1],
        # при noise_limits=[-0.00001, 0.00001],
        # при noise_limits=[-0.0001, 0.0001]
        # noise_limits=[-0.001, 0.001]
        # могут пропадать целые биты
        len_bar=ref_signal.ts,
        bar_colors_seed=bar_colors_seed
    )
    print(orig_bar_peaks_x, noise_bar_peaks_x)
    print(orig_bar_peaks_x == noise_bar_peaks_x)

    # Проведение квантизации
    quants = quantisation(opt_rec_orig_x, 1,
                          include_remains=True, where_include='center')
    print(f'{len(quants)=}')
    # print(quants[-1][-1] == opt_rec_orig_x[-1])

    binary_bits = generate_sequence(1)
    print(binary_bits)

    m_intv_distance = mean_interval_distance(quants, 1)
    print(f'{m_intv_distance=}')

    orig_peak_sequences = determine_peak_sequences(orig_bar_peaks_x, quants, binary_bits)
    print(f'{orig_peak_sequences}')

    noise_peak_sequences = determine_peak_sequences(noise_bar_peaks_x, quants, binary_bits)
    print(f'{noise_peak_sequences}')

    (all_equal_percents,
     mean_equal_percent) = comparing_bit_sequences(
        orig_peak_sequences, noise_peak_sequences)

    print(f'{all_equal_percents=}, {mean_equal_percent=}')

    plot_peaks_bar(orig_bar_peaks_x,
                   orig_bar_colors,
                   plot_axis=axes,
                   axis_num=0,
                   need_to_show_now=False,
                   height_bar=ref_signal.us,
                   width_bar=len(ref_signal.sig_x) * (ref_signal.sig_x[1] - ref_signal.sig_x[0]),
                   x_axis_label='Исходное расположение пиков',
                   y_axis_label='Высота пиков',
                   bit_seqs=orig_peak_sequences
                   )

    plot_peaks_bar(noise_bar_peaks_x,
                   noise_bar_colors,
                   plot_axis=axes,
                   axis_num=1,
                   need_to_show_now=False,
                   height_bar=ref_signal.us,
                   width_bar=len(ref_signal.sig_x) * (ref_signal.sig_x[1] - ref_signal.sig_x[0]),
                   x_axis_label='Сдвинутые шумом пики',
                   y_axis_label='Высота пиков',
                   bit_seqs=noise_peak_sequences
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

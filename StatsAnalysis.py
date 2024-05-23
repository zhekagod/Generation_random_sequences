from time import time

import numpy as np
import pandas as pd
import openpyxl
import xlsxwriter
import random as rnd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from GenData import ExtractSignal
from DataProcessing import \
    (quantisation, generate_sequence, mean_interval_distance,
     colored_print, gen_peaks_bar, determine_peak_sequences,
     comparing_bit_sequences, array_mean_extension, min_squares)


def extract_initial_signal_params(*args, **kwargs):
    reference_signal = ExtractSignal()
    if 'count_peaks' in kwargs and kwargs['count_peaks'] > 0:
        reference_signal.phys_channel.diff = kwargs['count_peaks']
    reference_signal.signal_extraction()
    opt_rec_original_x = reference_signal.opt_rec.orig_array_x
    len_opt_rec_original_x = len(opt_rec_original_x)
    # print(kwargs)
    quants_div = quantisation(opt_rec_original_x, **kwargs)
    bin_seq = generate_sequence(**kwargs)

    bar_colors_seed = rnd.randint(0, 256)
    orig_bar_peaks_x, orig_bar_colors, _ = gen_peaks_bar(
        reference_signal,
        len_bar=reference_signal.peaks_x,
        bar_colors_seed=bar_colors_seed,
        **kwargs)
    noise_bar_peaks_x, _, noise_value = gen_peaks_bar(
        reference_signal,
        add_noise=True,
        len_bar=reference_signal.peaks_x,
        bar_colors_seed=bar_colors_seed,
        **kwargs)
    noise_bar_colors = orig_bar_colors
    orig_peak_sequences = (
        determine_peak_sequences(orig_bar_peaks_x,
                                 quants_div, bin_seq))
    # print(noise_bar_peaks_x, quants_div, bin_seq, sep='\n')
    noise_peak_sequences = (
        determine_peak_sequences(noise_bar_peaks_x,
                                 quants_div, bin_seq))

    params = {'ref_signal': reference_signal,
              'opt_rec_orig_x': opt_rec_original_x,
              'len_opt_rec_orig_x': len_opt_rec_original_x,
              'quants': quants_div,
              'bin_seq': bin_seq,
              'orig_bar_peaks_x': orig_bar_peaks_x,
              'noise_bar_peaks_x': noise_bar_peaks_x,
              'bar_colors': noise_bar_colors,
              'bar_colors_seed': bar_colors_seed,
              'orig_peak_sequences': orig_peak_sequences,
              'noise_peak_sequences': noise_peak_sequences,
              'noise_value': noise_value}

    return params


def print_test_success_msg(func_name: str,
                           msg=' was successfully finished'):
    func_name = func_name.replace('t', 'T', 1)
    print(f'{func_name}{msg}')


def set_initial_test_progress():
    global progress_list
    for i in range(8):
        progress_list[i] = False


def print_test_progress_text(*args, **kwargs):
    global progress_list
    if kwargs['i'] >= round(kwargs['count_cycles'] * 0.95) and not \
            progress_list[7]:
        colored_print('Progress: 95%', color='green')
        progress_list[7] = True
        return
    elif kwargs['i'] >= round(kwargs['count_cycles'] * 0.9) and not \
            progress_list[6]:
        colored_print('Progress: 90%', color='green')
        progress_list[6] = True
        return
    elif kwargs['i'] >= round(kwargs['count_cycles'] * 0.8) and not \
            progress_list[5]:
        colored_print('Progress: 80%', color='green')
        progress_list[5] = True
        return
    elif kwargs['i'] >= round(kwargs['count_cycles'] * 0.6) and not \
            progress_list[4]:
        colored_print('Progress: 60%', color='green')
        progress_list[4] = True
        return
    elif kwargs['i'] >= round(kwargs['count_cycles'] * 0.4) and not \
            progress_list[3]:
        colored_print('Progress: 40%', color='green')
        progress_list[3] = True
        return
    elif kwargs['i'] >= round(kwargs['count_cycles'] * 0.3) and not \
            progress_list[2]:
        colored_print('Progress: 30%', color='green')
        progress_list[2] = True
        return
    elif kwargs['i'] >= round(kwargs['count_cycles'] * 0.1) and not \
            progress_list[1]:
        colored_print('Progress: 10%', color='green')
        progress_list[1] = True
        return
    elif kwargs['i'] >= round(kwargs['count_cycles'] * 0.01) and not \
            progress_list[0]:
        colored_print('Progress: 1%', color='green')
        progress_list[0] = True
        return


def test_mean_interval_distances(
        output_file='test_mean_interval_distances',
        gen_repeats=1024,
        noise_lims=None,
        file_type='.txt',
        file_mode='w+',
        include_skips=True,
        **kwargs):
    if noise_lims is None:
        noise_lims = [-0.0009866208565240113, 0.0009866208565240113]
    with open(output_file + file_type, file_mode) as output:
        start_time = time()
        skips = 0
        for i in range(gen_repeats):
            params = extract_initial_signal_params(bits_count=1024,
                                                   include_remains=True,
                                                   where_include='center',
                                                   noise_limits=noise_lims)
            if (not params['orig_peak_sequences']) or (not params['noise_peak_sequences']):
                skips += 1
                continue
            m_intv_distance = mean_interval_distance(params['quants'], 1024)
            # print(f'{m_intv_distance=}')
            output.write(str(m_intv_distance) + '\n')
            print_test_progress_text(i=i, count_cycles=gen_repeats)
            # print(i)
        else:
            end_time = time()
            delta_time = end_time - start_time
            mins = delta_time // 60
            secs = delta_time - mins * 60
            output.write(f'Test elapsed time: {mins} minutes and {secs} seconds')
            print(f'Test elapsed time: {mins} minutes and {secs} seconds')
            res = 0
            lines = output.readlines()
            for line in lines[:-1]:
                res += float(line)
            if include_skips:
                output.write(f'mean = {res / (len(lines[:-1]) + skips)}')
            else:
                output.write(f'mean = {res / (len(lines[:-1]))}')
    print_test_success_msg(test_mean_interval_distances.__name__)
    exit(0)


def test_mean_interval_distances_and_quants_level(
        output_file='test_mean_interval_distances_and_quants_level',
        count_cycles: int | list = 1000,
        gen_repeats=1000,
        noise_lims=None,
        file_type='.txt',
        file_mode='w+',
        include_skips=False,
        **kwargs):
    if noise_lims is None:
        noise_lims = [-0.0009866208565240113, 0.0009866208565240113]
    with open(output_file + file_type, file_mode) as output:
        start_time = time()
        if isinstance(count_cycles, int):
            cycle = range(count_cycles)
        elif isinstance(count_cycles, list):
            cycle = count_cycles
        quants_levels = []
        mean_distances = []
        skips = [0 for _ in range(len(cycle))]
        for i, level in enumerate(cycle):
            ms = 0
            for k in range(gen_repeats):
                params = extract_initial_signal_params(bits_count=level,
                                                       include_remains=True,
                                                       where_include='center',
                                                       noise_limits=noise_lims)
                if (not params['orig_peak_sequences']) or (not params['noise_peak_sequences']):
                    skips[i] += 1
                    continue
                m_intv_distance = mean_interval_distance(params['quants'], level)
                # print(f'{m_intv_distance=}')
                ms += m_intv_distance
                output.write(str(m_intv_distance) + ',' + str(level) + '\n')
                print_test_progress_text(i=i, count_cycles=len(cycle))
                # print(i)
            if skips[i] < 0.2 * gen_repeats:
                quants_levels.append(level)
                mean_distances.append(ms / gen_repeats)
        else:
            end_time = time()
            delta_time = end_time - start_time
            mins = delta_time // 60
            secs = delta_time - mins * 60
            output.write(f'Test elapsed time: {mins} minutes and {secs} seconds')
            print(f'Test elapsed time: {mins} minutes and {secs} seconds')
            res = sum(mean_distances)
            if include_skips:
                output.write(f'mean = {res / (len(mean_distances) + sum(skips))}')
            else:
                output.write(f'mean = {res / (len(mean_distances))}')
            df = pd.DataFrame()
            df['Уровень квантизации'] = quants_levels
            df['Среднее расстояние между интервалами'] = mean_distances
    print_test_success_msg(test_mean_interval_distances.__name__)
    return df


def test_quants_level_and_bit_error(
        output_file='test_quants_level_and_bit_error',
        count_cycles: int | list = 1024,
        gen_repeats=10000,
        noise_lims=None,
        file_type='.txt',
        file_mode='w+',
        **kwargs):
    if noise_lims is None:
        noise_lims = [-0.0009866208565240113, 0.0009866208565240113]
    with open(output_file + file_type, file_mode) as output:
        start_time = time()
        if isinstance(count_cycles, int):
            cycle = range(count_cycles)
        elif isinstance(count_cycles, list):
            cycle = count_cycles
        quants_levels = []
        errors = []
        skips = 0
        for i, level in enumerate(cycle):
            quants_level = level
            error = 0
            divider = gen_repeats
            for k in range(gen_repeats):
                params = extract_initial_signal_params(bits_count=quants_level,
                                                       include_remains=True,
                                                       where_include='center',
                                                       noise_limits=noise_lims)
                if (not params['orig_peak_sequences']) or (not params['noise_peak_sequences']):
                    skips += 1
                    continue
                mean_error = comparing_bit_sequences(
                    params['orig_peak_sequences'],
                    params['noise_peak_sequences'],
                    comparison_type=comparison_type)
                if (not mean_error[0]) and (mean_error[1] == 0):
                    divider -= 1
                error += mean_error[1]
            error /= divider
            if divider >= gen_repeats / 2:
                quants_levels.append(quants_level)
                errors.append(error)
            else:
                continue
            output.write(str(quants_level) + ',' + str(error) + '\n')
            print_test_progress_text(i=i, count_cycles=len(cycle))
        else:
            end_time = time()
            delta_time = end_time - start_time
            mins = delta_time // 60
            secs = delta_time - mins * 60
            output.write(f'Test elapsed time: {mins} minutes and {secs} seconds')
            print(f'Test elapsed time: {mins} minutes and {secs} seconds')
            df = pd.DataFrame()
            df['Уровень квантизации'] = quants_levels
            df['Долевая ошибка'] = errors
        print_test_success_msg(test_quants_level_and_bit_error.__name__)
        return df


def test_quants_level_and_noise_and_bit_error(
        output_file='test_quants_level_and_noise_and_bit_error',
        count_cycles: int | list = 1024,
        gen_repeats=10000,
        noise_limits_list=None,
        file_type='.txt',
        file_mode='w+',
        **kwargs):
    if noise_limits_list is None:
        raise AttributeError(f'Parameter "noise_limits_list" must '
                             f'not be equal to NoneType value, it '
                             f'should be list-like type of array')
    with open(output_file + file_type, file_mode) as output:
        start_time = time()
        if isinstance(count_cycles, int):
            cycle = range(count_cycles)
        elif isinstance(count_cycles, list):
            cycle = count_cycles
        quants_levels = []

        bit_errors = []
        noise_values = []
        skips = 0
        for i, level in enumerate(cycle):
            quants_level = level

            for noise_lims in noise_limits_list:
                bit_error = 0
                divider = gen_repeats
                noise = 0
                for k in range(gen_repeats):
                    params = extract_initial_signal_params(bits_count=quants_level,
                                                           include_remains=True,
                                                           where_include='center',
                                                           noise_limits=noise_lims,
                                                           **kwargs)
                    if (not params['orig_peak_sequences']) or (not params['noise_peak_sequences']):
                        skips += 1
                        continue
                    mean_error = comparing_bit_sequences(
                        params['orig_peak_sequences'],
                        params['noise_peak_sequences'],
                        **kwargs)
                    if (not mean_error[0]) and (mean_error[1] == 0):
                        divider -= 1
                    if 'comparison_type' in kwargs:
                        if kwargs['comparison_type'] == 'full_equal':
                            bit_error += mean_error[1]
                    noise += np.mean(params['noise_value'])
                bit_error /= divider
            if divider >= gen_repeats / 2:
                quants_levels.append(quants_level)
                bit_errors.append(bit_error)
                noise_values.append(noise / gen_repeats)
            else:
                continue
            output.write(str(quants_level) + ',' + str(bit_error) + '\n')
            print_test_progress_text(i=i, count_cycles=len(cycle))
        else:
            end_time = time()
            delta_time = end_time - start_time
            mins = delta_time // 60
            secs = delta_time - mins * 60
            output.write(f'Test elapsed time: {mins} minutes and {secs} seconds')
            print(f'Test elapsed time: {mins} minutes and {secs} seconds')
            df = pd.DataFrame()
            df['Уровень квантизации'] = quants_levels
            # df['Уровень шума'] = [noise_limits_list[i][1] for i in range(len(noise_limits_list))]
            df['Уровень шума'] = noise_values
            df['Процентная битовая ошибка'] = bit_errors

        print_test_success_msg(test_quants_level_and_bit_error.__name__)
        return df


if __name__ == '__main__':
    s_time = time()
    progress_list = [False for i in range(8)]
    # test_mean_interval_distances()
    '''
    with open('test_mean_interval_distances.txt', 'r') as f:
        lines = f.readlines()
        res = sum(map(float, lines[:-1]))/len(lines[:-1])
    print(res) # 0.0009866208565240113  
    '''
    '''
    cycle_range = [2 ** i for i in range(1, 13)]
    j = 1
    tmp = []
    for i in range(1, len(cycle_range)):
        tmp.append((cycle_range[i]+cycle_range[i-1])//2)
    # [3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072]
    for i in range(len(cycle_range) - 1):
        cycle_range.insert(i + j, tmp[i])
        j += 1
    print(cycle_range, tmp)
    '''
    # cycle_range = list(range(2, 100 + 1, 10))
    '''cycle_range = list(range(2, 100 + 1, 90))
    cycle_range.extend(range(cycle_range[-1], 1000, 100))
    cycle_range.append(1000)'''
    cycle_range = list(range(1, 11))
    print(cycle_range)
    noises_table = pd.read_excel(f'test_mean_interval_'
                                 f'distances_and_quants_level_'
                                 f'df4.xlsx')
    clm = 'Среднее расстояние между интервалами'
    '''
    noise_limits_list = [[-noises_table[clm][i], noises_table[clm][i]]
                         for i in range(len(noises_table))]
    # noise_limits_list = noise_limits_list[2:-2:2]
    noise_limits_list = [noise_limits_list[0]] + [noise_limits_list[9]] + \
                        [noise_limits_list[10]] + noise_limits_list[11::2]
    '''
    # count_peaks_list = [2, 3]
    # count_peaks_list += [i for i in range(5, 16, 2)]
    count_peaks_list = list(range(11, 16, 2))
    print(count_peaks_list)
    noise_limits_list = [[-0.5, 0.5] for _ in range(10)]
    print(noise_limits_list)
    print(len(cycle_range), len(noise_limits_list))
    print(f'=' * 100)
    for count_peaks in count_peaks_list:
        set_initial_test_progress()
        print(f'{count_peaks=}')

        '''m = np.linspace(2, 3, len(noise_limits_list)).tolist()
        print(m)'''
        '''noise_limits_list = [[noise_limits_list[i][0]*m[i], noise_limits_list[i][1]*m[i]]
                             for i in range(len(noise_limits_list))]'''

        '''array = [1, 2, 3, 4, 5, 6]
        extent = 5
        result = array_mean_extension(array, extent)
        print(result)'''
        '''df = test_quants_level_and_bit_error(count_cycles=cycle_range,
                                             gen_repeats=100)
        df.to_excel('test_quants_level_and_bit_error_df.xlsx', index=False)
        print(df)
        # plt.plot(df['Уровень квантизации'], df['Процентная ошибка'])
        # plt.show()'''

        df = test_quants_level_and_noise_and_bit_error(count_cycles=cycle_range,
                                                       gen_repeats=10,
                                                       noise_limits_list=noise_limits_list,
                                                       comparison_type='full_equal',
                                                       count_peaks=count_peaks)
        print(df)

        df.to_excel(f'test_q_level_noise_bit_error_equal_count_p{count_peaks}.xlsx', index=False)
        print(f'=' * 100)
    else:
        e_time = time()
        colored_print(f'All elapsed time is {(e_time - s_time) // 60} mins and '
                      f'{e_time - s_time - ((e_time - s_time) // 60) * 60} secs',
                      color='green')
        colored_print(f'All tests for count_peaks from 2 to 15 have done!')

    '''
   # рабочий код!!!
    
    df = pd.read_excel('test_quants_level_and_noise_and_bit_error_equal4.xlsx')
    # print(df)
    x, y, z = df['Уровень квантизации'].to_list(), df['Уровень шума'].to_list(), (df['Процентная битовая ошибка']*100).to_list()
    x = array_mean_extension(x, len(x) - 1)
    y = array_mean_extension(y, len(y) - 1)
    # y = np.linspace(0.494656565, 1.431324342, 25)
    z = array_mean_extension(z, len(z) - 1)

    y = array_mean_extension(y, len(y) - 1)
    y = array_mean_extension(y, len(y) - 1)
    y = array_mean_extension(y, len(y) - 1)
    #y *= 1.1
    #y[0:5] = np.linspace(0, 10.16666667, 5)
    print(x, y, z, sep='\n')


    # Создание сетки координат
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)


    # Повторение координат по одной из осей
    # Например, повторим координаты z по оси Y
    t1 = np.argmax(z)
    print(t1, len(Z))
    for i in range(len(z)):
        Z[i, :] = z

    print(Z)
    print(min_squares(x, y, z))
    a, b, c = min_squares(x, y, z)
    x_grid, y_grid = np.meshgrid(x, y)
    z_plane = a * x_grid + b * y_grid + c
    # Построение графика
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Построение линии
      
    x = array_mean_extension(x, len(x) - 1)
    x = array_mean_extension(x, len(x) - 1)
    x = array_mean_extension(x, len(x) - 1)
    z = array_mean_extension(z, len(z) - 1)
    z = array_mean_extension(z, len(z) - 1)
    z = array_mean_extension(z, len(z) - 1)
    
    ax.plot3D(x, y, z, 'red')
    ax.scatter3D(x, y, z, c=z, cmap='cividis')

    # Построение плоскости
    ax.plot_surface(x_grid, y_grid, Z, alpha=0.5, rstride=1, cstride=1)

    ax.set_xlabel('Уровень квантизации')
    ax.set_ylabel('Уровень шума')
    ax.set_zlabel('Процентое совпадение')
    # рабочий код!!!
    '''

    '''fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(x, y, z, 'red')
    ax.scatter3D(x, y, z, c=z, cmap='cividis')

    Y, X = np.meshgrid(np.linspace(min(y), max(y), len(y)), x)
    Z = np.zeros(X.shape)

    # Определяем шаг повторения линии
    num_repeats = 50
    z_min, z_max = min(z), max(z)
    Z_repeat = np.linspace(z_min, z_max, num_repeats)

    # Создание фигуры и 3D оси
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Повторение линии вдоль оси Z
    for z_val in Z_repeat:
        ax.plot3D(x, y, z_val + z - z[0], 'red')

    # Отображение исходных точек
    # ax.scatter3D(x, y, z, c=z, cmap='cividis')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')'''

    '''X, Y = np.meshgrid(x, y)
    z = np.meshgrid(X, Y)
    Z = np.zeros(X.shape)
    a, b, c = 1, 1, 0
    Z = a * X + b * Y + c
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    '''

    # ax.plot3D(x, y, z)

    # plt.show()
    # exit(0)

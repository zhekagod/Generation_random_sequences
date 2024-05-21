from time import time

import pandas as pd
import random as rnd
import matplotlib.pyplot as plt
from GenData import ExtractSignal
from DataProcessing import \
    (quantisation, generate_sequence, mean_interval_distance,
     colored_print, gen_peaks_bar, determine_peak_sequences,
     comparing_bit_sequences)


def extract_initial_signal_params(*args, **kwargs):
    reference_signal = ExtractSignal()
    reference_signal.signal_extraction()
    opt_rec_original_x = reference_signal.opt_rec.orig_array_x
    len_opt_rec_original_x = len(opt_rec_original_x)
    # print(kwargs)
    quants_div = quantisation(opt_rec_original_x, **kwargs)
    bin_seq = generate_sequence(**kwargs)

    bar_colors_seed = rnd.randint(0, 256)
    orig_bar_peaks_x, orig_bar_colors = gen_peaks_bar(
        reference_signal,
        len_bar=reference_signal.peaks_x,
        bar_colors_seed=bar_colors_seed,
        **kwargs)
    noise_bar_peaks_x = gen_peaks_bar(
        reference_signal,
        add_noise=True,
        noise_limits=[-0.0009866208565240113,
                      0.0009866208565240113],
        len_bar=reference_signal.peaks_x,
        bar_colors_seed=bar_colors_seed,
        **kwargs)[0]
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
              'noise_peak_sequences': noise_peak_sequences}

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
        count_cycles=1024,
        file_type='.txt',
        file_mode='w+'):
    with open(output_file + file_type, file_mode) as output:
        start_time = time()
        for i in range(count_cycles):
            params = extract_initial_signal_params(bits_count=1024,
                                                   include_remains=True,
                                                   where_include='center')
            m_intv_distance = mean_interval_distance(params['quants'], 1024)
            # print(f'{m_intv_distance=}')
            output.write(str(m_intv_distance) + '\n')
            print_test_progress_text(i=i, count_cycles=count_cycles)
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
            output.write(f'mean = {res / len(lines[:-1])}')
    print_test_success_msg(test_mean_interval_distances.__name__)
    exit(0)


def test_quants_level_and_bit_error(
        output_file='test_quants_level_and_bit_error',
        count_cycles: int | list = 1024,
        gen_repeats=10,
        file_type='.txt',
        file_mode='w+'):
    with open(output_file + file_type, file_mode) as output:
        start_time = time()
        if isinstance(count_cycles, int):
            cycle = range(count_cycles)
        elif isinstance(count_cycles, list):
            cycle = count_cycles
        quants_levels = []
        errors = []
        for i, level in enumerate(cycle):
            quants_level = level
            error = 0
            for k in range(gen_repeats):
                params = extract_initial_signal_params(bits_count=quants_level,
                                                       include_remains=True,
                                                       where_include='center')
                mean_error = comparing_bit_sequences(
                    params['orig_peak_sequences'],
                    params['noise_peak_sequences'])[1]
                error += mean_error
            error /= gen_repeats
            quants_levels.append(quants_level)
            errors.append(error)
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
            df['Процентная ошибка'] = errors
        print_test_success_msg(test_quants_level_and_bit_error.__name__)
        return df


if __name__ == '__main__':
    progress_list = [False for i in range(8)]
    set_initial_test_progress()
    # test_mean_interval_distances()
    '''
    with open('test_mean_interval_distances.txt', 'r') as f:
        lines = f.readlines()
        res = sum(map(float, lines[:-1]))/len(lines[:-1])
    print(res) # 0.0009866208565240113  
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
    df = test_quants_level_and_bit_error(count_cycles=cycle_range)
    print(df)
    plt.plot(df['Уровень квантизации'], df['Процентная ошибка'])
    plt.show()
    # exit(0)

from time import time
from GenData import ExtractSignal
from DataProcessing import \
    (quantisation, generate_sequence, mean_interval_distance,
     colored_print)


def extract_initial_signal_params(*args, **kwargs):
    reference_signal = ExtractSignal()
    reference_signal.signal_extraction()
    opt_rec_original_x = reference_signal.opt_rec.orig_array_x
    len_opt_rec_original_x = len(opt_rec_original_x)
    quants_div = quantisation(opt_rec_original_x, **kwargs)
    bin_seq = generate_sequence(**kwargs)
    params = {'ref_signal': reference_signal,
              'opt_rec_orig_x': opt_rec_original_x,
              'len_opt_rec_orig_x': len_opt_rec_original_x,
              'quants': quants_div,
              'bin_seq': bin_seq}

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
            mins = delta_time//60
            secs = delta_time - mins * 60
            output.write(f'Test elapsed time: {mins} minutes and {secs} seconds')
            print(f'Test elapsed time: {mins} minutes and {secs} seconds')
            res = 0
            lines = output.readlines()
            for line in lines[:-1]:
                res += float(line)
            output.write(f'mean = {res/len(lines[:-1])}')
    print_test_success_msg(test_mean_interval_distances.__name__)
    exit(0)


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
    exit(0)

"""To disable problems related to multithreading / forks in MacOS Sierra, please, add

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

to your .bash_profile

https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-
progr
"""

import argparse
import inspect
from time import sleep
from multiprocessing import Pool
from grampy.api import check, opc_check, upc_check
from grampy.text import AnnotatedTokens, AnnotatedText
from scorer.helpers.batcher import Batcher
from scorer.helpers.utils import read_lines, write_lines
from scorer.helpers.filtering import filter_by_error_type
from scorer.classifier.get_features import fix_patterns_output, fix_upc_output


def get_protected_response(check_func, params_dict):
    response = None
    while response is None:
        try:
            response = check_func(**params_dict)
        except:
            print('*** sleep happens!')
            sleep(10)
    if isinstance(response, list):
        response = response[0]
    return response


def get_combined_data(sentences, check_type, addr=None, filters=None,
                      custom_server=False):
    combined_data = [[x, check_type, addr, filters, custom_server]
                     for x in sentences]
    return combined_data


def wrapped_check_func(combined_data):
    # uncover combined data
    text, check_type, addr, filters, custom_server = combined_data

    # get custom parameters values
    custom_param_dict = {}
    if check_type == 'Patterns':
        check_func = check
        if addr is not None:
            custom_param_dict['patterns_addr'] = addr
    elif check_type.startswith('OPC'):
        check_func = opc_check
        if addr is not None:
            custom_param_dict['addr'] = addr
        custom_param_dict['filters'] = filters
    elif check_type.startswith('UPC'):
        check_func = upc_check
        if addr is not None:
            custom_param_dict['addr'] = addr
        custom_param_dict['custom_server'] = custom_server
    else:
        raise ValueError('Unknown check_type = %s' % check_type)

    # get defaults values
    func_parameters = inspect.getfullargspec(check_func).args
    func_defaults = inspect.getfullargspec(check_func).defaults
    assert len(func_defaults) == (len(func_parameters) - 1)
    params_dict = {}
    params_dict[func_parameters[0]] = text
    for i, def_value in enumerate(func_defaults):
        params_dict[func_parameters[i + 1]] = def_value

    # customize some values
    for key, value in custom_param_dict.items():
        params_dict[key] = value

    response = get_protected_response(check_func, params_dict)
    if check_type.startswith('UPC'):
        response = fix_upc_output(response)
    elif check_type == 'Patterns':
        response = fix_patterns_output(response)
    else:
        pass

    return response.get_annotated_text()


def run_check_parallel(orig_list, check_type, error_type, n_threads, fn_out):
    if check_type == 'Patterns':
        combined_data = get_combined_data(orig_list, check_type)
    elif check_type == 'OPC-with-filters':
        filters = {"<ErrorTypesFilter(types=None)>": {"types": [error_type]}}
        combined_data = get_combined_data(orig_list, check_type,
                                          addr='PREPROD', filters=filters)
    elif check_type == 'OPC-without-filters':
        filters = False
        combined_data = get_combined_data(orig_list, check_type,
                                          addr='PREPROD', filters=filters)
    elif check_type == 'UPC5-high-precision':
        combined_data = get_combined_data(orig_list, check_type)
    elif check_type == 'UPC5-high-recall':
        upc_addr = "upc-high-recall-server.phantasm.gnlp.io:8081"
        combined_data = get_combined_data(orig_list, check_type, addr=upc_addr,
                                          custom_server=True)
    else:
        raise ValueError('Unknown check_type = %s' % check_type)

    # create helper object to deal with batches
    batcher = Batcher(combined_data, batch_size=n_threads, verbose=True)
    pool = Pool(processes=n_threads) # pool to make multithreading
    result_anno = list()
    for batch in batcher.iter_batches():
        result_anno_batch = pool.map(wrapped_check_func, batch)
        result_anno.extend(result_anno_batch)
    pool.close()
    pool.join()
    # Normalizing trick
    normalized_result_anno = [AnnotatedTokens(AnnotatedText(x)).get_annotated_text() for x in result_anno]
    write_lines(fn_out, normalized_result_anno)
    return normalized_result_anno


def main(args):
    # Read original texts
    test_orig = read_lines(args.test_orig)
    # Run checks in parallel and save result
    out_file = args.test_orig.replace('.txt', f'_{args.system_type}.txt')
    run_check_parallel(test_orig, check_type=args.system_type,
                       error_type=args.error_type, n_threads=args.n_threads,
                       fn_out=out_file)
    # Filter output
    unfiltered_data = read_lines(out_file)
    output = filter_by_error_type(unfiltered_data,
                                  error_type=args.error_type,
                                  system_type=args.system_type)
    # Save results
    out_filtered_file = out_file.replace('.txt', f'_by_{args.error_type}.txt')
    write_lines(out_filtered_file, output)


if __name__ == '__main__':
    print('Step 2 / 3, run external services and write their output to text '
          'files.')
    parser = argparse.ArgumentParser()
    parser.add_argument('test_orig',
                        help='Path to the input txt file with original '
                             'input texts')
    parser.add_argument('--n_threads',
                        type=int,
                        help='Batch size, it is equal to number of threads.',
                        default=100)
    parser.add_argument('--system_type',
                        help='Type of the system for calculations.',
                        choices=['Patterns',
                                 'OPC-with-filters',
                                 'OPC-without-filters',
                                 'UPC5-high-precision',
                                 'UPC5-high-recall'],
                        default='UPC5-high-recall')
    parser.add_argument('--error_type',
                        help='The only type which you want to select.',
                        default='Pronoun')
    args = parser.parse_args()
    main(args)
    print('Finished step 2 / 3.')

"""Script for filtering file by error type"""
import argparse
from scorer.helpers.utils import read_lines, write_lines
from scorer.step_2_run_corr_service import filter_by_error_type


def main(args):
    # Read original texts
    unfiltered_data = read_lines(args.unfiltered_file)
    # Filter text
    output, cnt = filter_by_error_type(unfiltered_data,
                                       error_type=args.error_type,
                                       system_type=args.system_type)
    # Save results
    out_file = args.unfiltered_file.replace('.txt', f'_by_{args.error_type}.txt')
    write_lines(out_file, output)


if __name__ == '__main__':
    print('Filter files.')
    parser = argparse.ArgumentParser()
    parser.add_argument('unfiltered_file',
                        help='Path to the input txt file with all annotations.')
    parser.add_argument('--error_type',
                        help='The only type which you want to select.',
                        default='Determiner')
    parser.add_argument('--system_type',
                        help='Type of the system for understanding error '
                             'typology.',
                        choices=['UPC', 'OPC', 'Patterns', 'CLC'],
                        default='CLC')
    args = parser.parse_args()
    main(args)

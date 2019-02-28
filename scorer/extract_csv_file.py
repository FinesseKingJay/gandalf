"""Script for filtering file by error type"""
import argparse
from scorer.helpers.clc_csv_reader import ClcCsvReader
from scorer.helpers.utils import write_lines

def main(args):
    # Read original texts
    clc_reader = ClcCsvReader(args.input_file)
    output = []
    for _, _, _, _, relabeled_anno_text, _ in clc_reader.iter_items():
        output.append(relabeled_anno_text)

    write_lines(args.output_file, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        help='Path to the input txt file with all annotations.')
    parser.add_argument('output_file',
                        help='Path to the input txt file with all annotations.')
    args = parser.parse_args()
    main(args)

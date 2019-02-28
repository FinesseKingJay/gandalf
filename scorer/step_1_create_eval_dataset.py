"""To get the CLC-test dataset, type:

 scp your_firstname.your_lastname@etc-punct.phantasm.gnlp.io://corpora/CLC/CLC-full/CLC-test.csv .

 CLC dataset description: https://docs.google.com/document/d/1qvULEovmx2GhzFlzw5ZgMezbHxDuEu_t_gBCoUFCSdY
 """

import argparse
from scorer.helpers.clc_csv_reader import ClcCsvReader
from scorer.helpers.error_types_bank import ErrorTypesBank
from scorer.helpers.filtering import filter_by_nosuggestions_in_gold
from scorer.helpers.utils import write_lines, is_lists_intersection
from grampy.text import AnnotatedTokens, AnnotatedText


def main(args):
    clc_csv_reader = ClcCsvReader(fn=args.fn_clc_csv)
    error_types_bank = ErrorTypesBank()
    target_error_types_list = error_types_bank.patterns22_to_clc89(args.target_error_type)
    orig_lines = list() # original texts
    gold_annotations = list() # gold corrections in Annotated Text string format
    for _, _, _, _, gold_relabeled_anno_text, gold_error_types_list \
            in clc_csv_reader.iter_items(max_item_number=args.max_item_number):
        # We are not interested in the text samples which doesn''t contain
        # at least one target error type
        if not is_lists_intersection(gold_error_types_list,
                                     target_error_types_list):
            continue
        ann_tokens = AnnotatedTokens(AnnotatedText(gold_relabeled_anno_text))
        for ann in ann_tokens.iter_annotations():
            if ann.meta['error_type'] not in target_error_types_list:
                ann_tokens.remove(ann)
        gold_annotations_renormalized = ann_tokens.get_annotated_text()
        # Add renormalized texts to the lists
        orig_sent = ann_tokens.get_original_text()
        orig_lines.append(orig_sent)
        gold_annotations.append(gold_annotations_renormalized)
    assert len(orig_lines) == len(gold_annotations)
    print('%d lines in unfiltered outputs.' % len(orig_lines))
    gold_annotations_filtered, orig_lines_filtered = filter_by_nosuggestions_in_gold(gold_annotations, orig_lines)
    assert len(gold_annotations_filtered) == len(orig_lines_filtered)
    print('%d lines in filtered by NO_SUGGESTION flag outputs.' % len(orig_lines_filtered))
    # Write to files
    fn_out_gold_file = args.fn_clc_csv.replace('.csv', f'_{args.target_error_type}_gold.txt')
    fn_out_orig_file = args.fn_clc_csv.replace('.csv', f'_{args.target_error_type}_orig.txt')
    write_lines(fn=fn_out_gold_file, lines=gold_annotations_filtered)
    write_lines(fn=fn_out_orig_file, lines=orig_lines_filtered)


if __name__ == '__main__':
    print('Step 1 / 3, create eval dataset.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn_clc_csv',
                        help='Path to the CLC-test input file',
                        default='CLC-test.csv')
    parser.add_argument('--target_error_type',
                        help='Target error type from Patterns22',
                        default='Determiner')
    parser.add_argument('--max_item_number',
                        type=int,
                        help='Maximum number of items to get, set 0 to get all '
                             'items',
                        default=None)
    args = parser.parse_args()
    main(args)
    print('Finished step 1 / 3.')

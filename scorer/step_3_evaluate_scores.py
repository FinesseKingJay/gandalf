import argparse
from os import system
import os
from grampy.text import AnnotatedTokens, AnnotatedText
from grampy.text.m2 import M2Annotation, MultiAnnotatedSentence
from scorer.helpers.utils import read_lines, write_lines
from scorer.helpers.filtering import filter_by_error_type
from scorer.helpers.error_types_bank import ErrorTypesBank


def remove_file(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


def evaluate_from_m2_file(m2_file, sub_lines, tmp_filename):
    output_corrected = [AnnotatedText(x).get_corrected_text() for x in sub_lines]
    sub_file_processed = f"o_{os.path.basename(tmp_filename)}"
    write_lines(sub_file_processed, output_corrected)
    m2_path = os.path.join(os.getcwd().split("/gandalf/")[0],
                           "gandalf/scorer/m2scorer/m2scorer")
    system(f'{m2_path} {sub_file_processed} {m2_file}')
    remove_file(sub_file_processed)


def evaluate_with_m2(gold_annotations, output_annotations, tmp_filename):
    assert len(gold_annotations) == len(output_annotations)
    gold_ann_tokens = [AnnotatedTokens(AnnotatedText(anno_text)) for anno_text in gold_annotations]
    gold_m2_annotations = []
    for ann_tokens in gold_ann_tokens:
        try:
            converted = MultiAnnotatedSentence.from_annotated_tokens(ann_tokens).to_m2_str() + '\n'
            gold_m2_annotations.append(converted)
        except Exception as e:
            # print(e)
            # print(ann_tokens.get_original_text())
            # print(ann_tokens.get_annotated_text())
            # print(ann_tokens.get_annotated_text(with_meta=False))
            # new_ann_tokens = AnnotatedTokens(ann_tokens._tokens)
            for ann in ann_tokens.iter_annotations():
                if not ann.suggestions or str(ann.suggestions[0]) == "NO_SUGGESTIONS":
                    ann_tokens.remove(ann)
            new_converted = MultiAnnotatedSentence.from_annotated_tokens(ann_tokens).to_m2_str() + '\n'
            gold_m2_annotations.append(new_converted)

    output_corrected_texts = [AnnotatedText(anno_text).get_corrected_text() for anno_text in output_annotations]
    # Write as text files

    gold_file_processed = f"g_{os.path.basename(tmp_filename)}"
    sub_file_processed = f"o_{os.path.basename(tmp_filename)}"
    write_lines(gold_file_processed, gold_m2_annotations)
    write_lines(sub_file_processed, output_corrected_texts)
    # Run m2scorer (OFFICIAL VERSION 3.2, http://www.comp.nus.edu.sg/~nlp/conll14st.html)
    system(f'./m2scorer/m2scorer {sub_file_processed} {gold_file_processed}')
    remove_file(sub_file_processed)
    remove_file(gold_file_processed)


def calculate_fp_ratio(lines):
    total_lines = len(lines)
    errors = sum([len(AnnotatedText(line).get_annotations()) for line in lines])
    fp_ratio = errors/max(total_lines, 1)
    print(f"FP ratio is {round(fp_ratio, 5)*100}%; found {errors} errors")


def main(args):
    # evaluate for all errors
    print(f"{args.fn_output}")
    print("Evaluate on all errors")
    gold_lines = read_lines(args.fn_gold)
    sub_lines = read_lines(args.fn_output)
    tmp_filename = args.fn_output.replace(".txt", "_tmp.txt")
    if args.fp_ratio:
        calculate_fp_ratio(sub_lines)
    else:
        if args.fn_gold.endswith(".m2"):
            evaluate_from_m2_file(args.fn_gold, sub_lines, tmp_filename)
            exit()
        else:
            evaluate_with_m2(gold_lines, sub_lines, tmp_filename)
    # evaluate on all categories

    eb = ErrorTypesBank()
    categories = eb.get_error_types_list("Patterns22")
    categories = ['Agreement', 'Pronoun', 'Punctuation', 'Preposition', 'Determiner', 'VerbSVA']
    categories = []
    for category in categories:
        print(f"\nEvaluate on {category}")
        gold_filtered, n_errors_in_gold = filter_by_error_type(gold_lines,
                                                               category, "CLC")
        sub_filtered, n_errors_in_sub = filter_by_error_type(sub_lines, category)
        if not n_errors_in_sub:
            print(f"There is no errors of {category} in sub")
            continue
        if args.fp_ratio:
            calculate_fp_ratio(sub_filtered)
        else:
            evaluate_with_m2(gold_filtered, sub_filtered, tmp_filename)


if __name__ == '__main__':
    print('Step 3 / 3, evaluate m2 scores.')
    parser = argparse.ArgumentParser()
    parser.add_argument('fn_gold',
                        help='Path to the gold input txt file')
    parser.add_argument('fn_output',
                        help='Path to the output txt file')
    parser.add_argument('--fp_ratio',
                        action='store_true',
                        help='Set if you want only calculate FP ratio')
    args = parser.parse_args()
    main(args)
    print('Finished step 3 / 3.')

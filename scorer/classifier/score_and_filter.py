import argparse
import requests
import json
import sys
import os
import random
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from grampy.text import AnnotatedText, AnnotatedTokens

from scorer.helpers.utils import read_lines, write_lines
from scorer.classifier.get_features import get_protected_response, \
    get_normalized_error_type
from scorer.combine_systems import get_kenlm_scores
from scorer.step_3_evaluate_scores import evaluate_from_m2_file, remove_file


def wrap_check(comb_data):
    sent, system_type = comb_data
    return get_protected_response(sent, system_type=system_type,
                                  error_type=None)


def get_confidence_score(batch, addr):
    r = requests.post(addr, json=batch)
    response = json.loads(r.text)
    return response


def wrap_confidence_scorer(combined):
    sent, addr = combined
    if not sent:
        return sent
    processed_sent = ""
    while not processed_sent:
        try:
            processed_sent = get_confidence_score([sent], addr)[0]
        except Exception as e:
            print(f"Something went wrong with confidence scoring. "
                  f"Exception which was raised {e}. Sleep for 5 sec")
            sleep(5)
    return processed_sent


def wrap_get_lm_scores(sent):
    ann_sent = ""
    if not sent:
        return sent
    while not ann_sent:
        try:
            ann_tokens = AnnotatedTokens(AnnotatedText(sent))
            for ann in ann_tokens.iter_annotations():
                scores = get_kenlm_scores(ann_tokens, [ann], add_original=True)
                confidence = scores[0] - scores[1]
                ann.meta['confidence'] = confidence
            ann_sent = ann_tokens.get_annotated_text()
        except Exception as e:
            print(f"Something wrong with KenLM. Exception {e}. Sentence {sent}")
    return ann_sent


def get_lines_from_m2_file(m2_file):
    tmp_number = int(random.random()*10**9)
    tmp_file = m2_file.replace(".m2", f"_tmp_{tmp_number}.txt")
    os.system(f'cat {m2_file} | grep "^S " | cut -c3- > {tmp_file}')
    lines = read_lines(tmp_file)
    remove_file(tmp_file)
    return lines


def main(args):
    all_files = os.listdir(args.input_dir)
    for fname in all_files:
        print(f"Start evaluation {args.system_type} on {fname}")
        if fname.endswith(".txt"):
            fp_ratio = True
        elif fname.endswith(".m2"):
            fp_ratio = False
        else:
            continue
        input_file = os.path.join(args.input_dir, fname)
        if fp_ratio:
            sentences = read_lines(input_file)
        else:
            sentences = get_lines_from_m2_file(input_file)

        # run through system
        if args.system_type is not None:
            combined = [(x, args.system_type) for x in sentences]
            with ThreadPoolExecutor(args.n_threads) as pool:
                system_out = list(tqdm(pool.map(wrap_check, combined),
                                    total=len(combined)))
            system_out = [x.get_annotated_text() for x in system_out]
        else:
            system_out = sentences
        print("System response was got")

        # run system through confidence scorer
        for scorer in [None, "LM", "CLF"]:
            print(f"Current scorer is {scorer}")
            if scorer == "CLF":
                combined = [(x, args.server_path) for x in system_out]
                with ThreadPoolExecutor(args.n_threads) as pool:
                    scorer_out = list(tqdm(pool.map(wrap_confidence_scorer, combined),
                                           total=len(combined)))
                thresholds = [0.1, 0.2, 0.25, 0.3, 0.5]
            elif scorer == "LM":
                with ThreadPoolExecutor(args.n_threads) as pool:
                    scorer_out = list(tqdm(pool.map(wrap_get_lm_scores, system_out),
                                           total=len(combined)))
                thresholds = [0]
            else:
                scorer_out = system_out
                thresholds = [None]
            print("Scores were got")

            # apply thresholds
            if args.error_types is not None:
                error_types = args.error_types.split()
            else:
                error_types = None
            for t in thresholds:
                t_out = []
                for sent in scorer_out:
                    ann_sent = AnnotatedTokens(AnnotatedText(sent))
                    for ann in ann_sent.iter_annotations():
                        ann.meta['system_type'] = args.system_type
                        et = get_normalized_error_type(ann)
                        if error_types is not None and et not in error_types:
                            ann_sent.remove(ann)
                            continue
                        score = float(ann.meta.get('confidence', 1))
                        if t is not None and score < t:
                            ann_sent.remove(ann)
                    t_out.append(ann_sent.get_annotated_text())
                if fp_ratio:
                    cnt_errors = sum([len(AnnotatedText(x).get_annotations()) for x in t_out])
                    print(f"\nThe number of errors are equal {cnt_errors}. "
                          f"FP rate {round(100*cnt_errors/len(t_out),2)}%")
                else:
                    print(f"\nThreshold level is {t}")
                    tmp_filename = input_file.replace(".m2", f"_{args.system_type}_{scorer}_above_{t}_tmp.txt")
                    evaluate_from_m2_file(input_file, t_out, tmp_filename)


if __name__ == "__main__":
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir',
                        help='Path to the directory with m2 files',
                        )
    parser.add_argument('--server_path',
                        help='Path to the server',
                        # default="http://0.0.0.0:8081/process"
                        default="http://opc-scorer.phantasm.gnlp.io:8081/process"
                        )
    parser.add_argument('--system_type',
                        help='Specify which system you want to try',
                        choices=['OPC', 'OPC-filtered', 'UPC', None],
                        default=None)
    parser.add_argument('--error_types',
                        help='Set if you want to filter errors by types.',
                        default=None)
    parser.add_argument('--n_threads',
                        help='Specify how many threads you want to use.',
                        type=int,
                        default=100)
    args = parser.parse_args()
    code = main(args)
    sys.exit(code)

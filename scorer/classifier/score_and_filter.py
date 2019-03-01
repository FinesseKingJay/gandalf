import argparse
import requests
import json
import sys
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from grampy.text import AnnotatedText, AnnotatedTokens

from scorer.helpers.utils import read_lines, write_lines
from scorer.classifier.get_features import get_protected_response, \
    get_normalized_error_type


def wrap_opc(sent):
    return get_protected_response(sent, system_type="OPC", error_type=None)


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


def main(args):
    # load sentences from input file
    sentences = read_lines(args.input_file)

    # run sentences through OPC if it needed
    if args.opc:
        with ThreadPoolExecutor(args.n_threads) as pool:
            opc_out = list(tqdm(pool.map(wrap_opc, sentences),
                                total=len(sentences)))
        opc_out = [x.get_annotated_text() for x in opc_out]
        out_file = args.output_file.replace(".txt", f"_opc.txt")
        write_lines(out_file, opc_out)
    else:
        opc_out = sentences
    print("OPC data was got")

    # run system through confidence scorer
    if args.score:
        combined = [(x, args.server_path) for x in opc_out]
        with ThreadPoolExecutor(args.n_threads) as pool:
            scorer_out = list(tqdm(pool.map(wrap_confidence_scorer, combined),
                                   total=len(combined)))
        out_file = args.output_file.replace(".txt", f"_scored.txt")
        write_lines(out_file, scorer_out)
    else:
        scorer_out = opc_out
    print("Scores were got")

    # apply thresholds
    thresholds = [0, 0.3, 0.5, 0.7]
    if args.error_types is not None:
        error_types = args.error_types.split()
    else:
        error_types = None
    for t in thresholds:
        t_out = []
        for sent in scorer_out:
            ann_sent = AnnotatedTokens(AnnotatedText(sent))
            for ann in ann_sent.iter_annotations():
                ann.meta['system_type'] = "OPC"
                et = get_normalized_error_type(ann)
                if error_types is not None and et not in error_types:
                    ann_sent.remove(ann)
                    continue
                score = float(ann.meta['confidence'])
                if score < t:
                    ann_sent.remove(ann)
            t_out.append(ann_sent.get_annotated_text())
        out_file = args.output_file.replace(".txt", f"_above_{t}.txt")
        write_lines(out_file, t_out)


if __name__ == "__main__":
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        help='Path to the input file',
                        )
    parser.add_argument('output_file',
                        help='Path to the output file',
                        )
    parser.add_argument('--server_path',
                        help='Path to the server',
                        # default="http://0.0.0.0:8081/process"
                        default="http://opc-scorer.phantasm.gnlp.io:8081/process"
                        )
    parser.add_argument('--opc',
                        action='store_true',
                        help='If set then data should be run through opc',
                        default=False)
    parser.add_argument('--score',
                        action='store_true',
                        help='If set then data should be run through scorer',
                        default=False)
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

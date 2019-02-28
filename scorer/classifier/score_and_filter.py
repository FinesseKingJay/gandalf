import argparse
import requests
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from grampy.api import opc_check
from grampy.text import AnnotatedText, AnnotatedTokens

from scorer.helpers.utils import read_lines, write_lines
from scorer.classifier.get_features import get_protected_response


def wrap_opc(sent):
    return get_protected_response(sent, system_type="OPC", error_type=None)


def get_confidence_score(batch, addr):
    r = requests.post(addr, json=batch)
    response = json.loads(r.text)
    return response


def wrap_confidence_scorer(combined):
    sent, addr = combined
    processed_sent = get_confidence_score([sent], addr)[0]
    return processed_sent


def main(args):
    # load sentences from input file
    sentences = read_lines(args.input_data)

    # run sentences through OPC if it needed
    if args.opc:
        with ThreadPoolExecutor(args.n_threads) as pool:
            opc_out = pool.map(wrap_opc, sentences)
        opc_out = [x for x in opc_out]
        out_file = args.output_file.replace(".txt", f"_opc.txt")
        write_lines(out_file, opc_out)
    else:
        opc_out = sentences

    # run system through confidence scorer
    combined = [(x, args.server_path) for x in opc_out]
    with ThreadPoolExecutor(args.n_treads) as pool:
        scorer_out = pool.map(wrap_confidence_scorer, combined)
    scorer_out = [x for x in scorer_out]
    out_file = args.output_file.replace(".txt", f"_scored.txt")
    write_lines(out_file, scorer_out)
    
    # apply thresholds
    thresholds = [0, 0.3, 0.5, 0.7]
    for t in thresholds:
        t_out = []
        for sent in scorer_out:
            ann_sent = AnnotatedTokens(AnnotatedText(sent))
            for ann in ann_sent.iter_annotations():
                score = ann.meta['confidence']
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

    args = parser.parse_args()
    code = main(args)
    sys.exit(code)

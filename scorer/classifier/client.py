import argparse
from collections import ChainMap
import requests
import json
from json.decoder import JSONDecodeError
import sys

from grampy.api.capi import tokenize
from grampy.api import opc_check
from grampy.text import AnnotatedTokens, AnnotatedText


def send_request(request, uri):
    r = requests.post(uri, json=request)
    response = json.loads(r.text)
    return response


def main(args):
    sentences = ["He go at school !",
                 "Worked in the oil business , started my own .",
                 ]
    # get opc responses
    ann_sents = [opc_check(x, addr="PREPROD", filters=False) for x in sentences]
    batch = [x.get_annotated_text() for x in ann_sents]
    print("Got output from OPC")

    # get confidenece scores
    results = send_request(batch, args.server_path)
    for sent, rez in zip(batch, results):
        print(sent)
        print(rez)
        print('\n')
    return 0


if __name__ == "__main__":
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_path',
                        help='Path to the server',
                        # default="http://0.0.0.0:8081/process"
                        default="http://opc-scorer.phantasm.gnlp.io:8081/process"
                        )

    args = parser.parse_args()
    code = main(args)
    sys.exit(code)

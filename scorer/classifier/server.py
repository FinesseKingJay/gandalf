import argparse
from flask import Flask, request
import waitress
import json
import numpy as np

from grampy.text import AnnotatedText, AnnotatedTokens

from scorer.classifier.classifier import load_models, get_all_indep_features, \
    get_clf_pred_probs, add_new_features
from scorer.classifier.get_features import get_features

app = Flask(__name__)


def preprocess_batch(batch, system_type):
    records = []
    ann_sents_dict = {}
    for i, sent in enumerate(batch):
        ann_sent = AnnotatedTokens(AnnotatedText(sent))
        ann_sents_dict[i] = ann_sent
        for ann in ann_sent.iter_annotations():
            ann.meta["system_type"] = system_type
            records.append([ann_sent, ann, (i, ann.start, ann.end)])
    return records, ann_sents_dict


def make_predictions(clf, scaler, selector, records):
    features_names = get_all_indep_features()
    scores = []
    for [ann_sent, ann, pos] in records:
        features_dict = get_features(ann_sent, ann, [ann], [ann])
        fd = add_new_features(features_dict)
        features_dict = {**features_dict, **fd}
        features = np.array([float(features_dict[x]) for x in features_names])
        features_norm = scaler.transform(features.reshape(1, -1))
        features_selected = selector.transform(features_norm)
        pred_probs = get_clf_pred_probs(clf, features_selected)
        scores.append([pos, pred_probs])
    return scores


def postprocess_sents(ann_sents_dict, scores):
    for [pos, score] in scores:
        id, start, end = pos
        ann_sents_dict[id].get_annotation_at(start, end).meta['confidence'] = score[0][1]
        _ = ann_sents_dict[id].get_annotation_at(start, end).meta.pop('system_type')
    output_batch = list([x.get_annotated_text() for x in ann_sents_dict.values()])
    return output_batch


def handle_batch(clf, scaler, selector, batch):
    if isinstance(batch, dict) and 'system_type' in batch.keys():
        system_type = batch.get('system_type')
        texts = batch.get('texts')
    else:
        system_type = "OPC"
        texts = batch
    records, ann_sents_dict = preprocess_batch(texts, system_type)
    scores = make_predictions(clf, scaler, selector, records)
    output_batch = postprocess_sents(ann_sents_dict, scores)
    return output_batch


@app.route('/process', methods=['POST'])
def process_request():
    batch = request.json
    if not batch:
        return json.dumps(batch)
    else:
        try:
            response = handle_batch(clf, scaler, selector, batch)
            response = json.dumps(response)
            return response
        except Exception as ex:
            print("Oops! Something bad happened. This is the request: %s" % batch)
            print(ex)
            raise


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path',
                        help='Path with the model file name')
    parser.add_argument('--port',
                        type=int,
                        default=8081)
    args = parser.parse_args()
    clf, scaler, selector = load_models(args.model_path)
    print("Server is running")
    waitress.serve(app, port=args.port, host="0.0.0.0")

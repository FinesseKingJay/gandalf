import argparse
import numpy as np
import pickle
import os
from random import random, shuffle
from copy import deepcopy
from tqdm import tqdm
from grampy.text import AnnotatedText, AnnotatedTokens
from scorer.helpers.utils import read_lines, write_lines
from scorer.classifier.get_features import get_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from concurrent.futures import ThreadPoolExecutor
from scorer.classifier.get_features import get_normalized_error_type


def get_processed_records(train_file, error_types=[],
                          store_sents=True, skip_dep_features=False):
    all_ann_sents = []
    features, labels = [], []
    error_types_list = []
    features_names = get_full_list_of_features()
    if skip_dep_features:
        dep_features = get_list_of_dependent_features()
        features_names = [x for x in features_names if x not in dep_features]
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            sent = line.strip()
            ann_sent = AnnotatedTokens(AnnotatedText(sent))
            for ann in ann_sent.iter_annotations():
                # apply error_type features
                ann_error_type = get_normalized_error_type(ann)
                if error_types and ann_error_type not in error_types:
                        ann_sent.remove(ann)
                        continue
                # check if ann has all features
                absent_features = [x for x in features_names
                                   if x not in list(ann.meta.keys())]
                if not absent_features:
                    # process annotation
                    try:
                        tmp_x, label, ordered_features_names = \
                            process_record(ann.meta, features_names)
                        features.append(tmp_x[:])
                        labels.append(label)
                        error_types_list.append(ann_error_type)
                    except Exception as e:
                        print(ann.meta)
                        continue
                else:
                    ann_sent.remove(ann)
            if store_sents:
                all_ann_sents.append(ann_sent)
    features_names = ordered_features_names
    features = np.array(features)
    labels = np.array(labels)
    return features, labels, features_names, error_types_list, all_ann_sents


def get_all_records(train_file, error_types=[], store_sents=True):
    all_sents = read_lines(train_file)
    print("All sents are loaded")
    all_records = []
    all_ann_sents = []
    for i, sent in tqdm(enumerate(all_sents)):
        ann_sent = AnnotatedTokens(AnnotatedText(sent))
        for ann in ann_sent.iter_annotations():
            # apply error_type features
            if error_types:
                ann_error_type = get_normalized_error_type(ann)
                if ann_error_type not in error_types:
                    ann_sent.remove(ann)
                    continue
            # check if ann has features (dict should be big)
            if len(ann.meta.keys()) > 5:
                all_records.append(ann.meta)
            # remove ann if there are not big dict there
            else:
                ann_sent.remove(ann)
        if store_sents:
            all_ann_sents.append(ann_sent)
    return all_records, all_ann_sents


def get_full_list_of_features():
    return ['Adjective_is_error_type_of_this_ann',
             'Adverb_is_error_type_of_this_ann',
             'Agreement_is_error_type_of_this_ann',
             'Conjunction_is_error_type_of_this_ann',
             'Determiner_is_error_type_of_this_ann',
             'Enhancement_is_error_type_of_this_ann',
             'Fluency_is_error_type_of_this_ann',
             'Morphology_is_error_type_of_this_ann',
             'Negation_is_error_type_of_this_ann',
             'Noun_is_error_type_of_this_ann',
             'OtherError_is_error_type_of_this_ann',
             'Preposition_is_error_type_of_this_ann',
             'Pronoun_is_error_type_of_this_ann',
             'Punctuation_is_error_type_of_this_ann',
             'Register_is_error_type_of_this_ann',
             'SentenceBoundary_is_error_type_of_this_ann',
             'SpellConfused_is_error_type_of_this_ann',
             'Spelling_is_error_type_of_this_ann',
             'Style_is_error_type_of_this_ann',
             'VerbSVA_is_error_type_of_this_ann',
             'VerbTense_is_error_type_of_this_ann',
             'Verb_is_error_type_of_this_ann',
             'WordOrder_is_error_type_of_this_ann',
             'is_confused',
             'is_delete',
             'is_insert',
             'is_opc',
             'is_patterns',
             'is_upc',
             'kenlm_ann_score',
             'kenlm_diff_score',
             'kenlm_orig_score',
             'len_tokens',
             'n_Adjective_anns_are_here',
             'n_Adjective_anns_are_in_other_places',
             'n_Adverb_anns_are_here',
             'n_Adverb_anns_are_in_other_places',
             'n_Agreement_anns_are_here',
             'n_Agreement_anns_are_in_other_places',
             'n_Conjunction_anns_are_here',
             'n_Conjunction_anns_are_in_other_places',
             'n_Determiner_anns_are_here',
             'n_Determiner_anns_are_in_other_places',
             'n_Enhancement_anns_are_here',
             'n_Enhancement_anns_are_in_other_places',
             'n_Fluency_anns_are_here',
             'n_Fluency_anns_are_in_other_places',
             'n_Morphology_anns_are_here',
             'n_Morphology_anns_are_in_other_places',
             'n_Negation_anns_are_here',
             'n_Negation_anns_are_in_other_places',
             'n_Noun_anns_are_here',
             'n_Noun_anns_are_in_other_places',
             'n_OtherError_anns_are_here',
             'n_OtherError_anns_are_in_other_places',
             'n_Preposition_anns_are_here',
             'n_Preposition_anns_are_in_other_places',
             'n_Pronoun_anns_are_here',
             'n_Pronoun_anns_are_in_other_places',
             'n_Punctuation_anns_are_here',
             'n_Punctuation_anns_are_in_other_places',
             'n_Register_anns_are_here',
             'n_Register_anns_are_in_other_places',
             'n_SentenceBoundary_anns_are_here',
             'n_SentenceBoundary_anns_are_in_other_places',
             'n_SpellConfused_anns_are_here',
             'n_SpellConfused_anns_are_in_other_places',
             'n_Spelling_anns_are_here',
             'n_Spelling_anns_are_in_other_places',
             'n_Style_anns_are_here',
             'n_Style_anns_are_in_other_places',
             'n_VerbSVA_anns_are_here',
             'n_VerbSVA_anns_are_in_other_places',
             'n_VerbTense_anns_are_here',
             'n_VerbTense_anns_are_in_other_places',
             'n_Verb_anns_are_here',
             'n_Verb_anns_are_in_other_places',
             'n_WordOrder_anns_are_here',
             'n_WordOrder_anns_are_in_other_places',
             'parsed_diff',
             'parsed_prob_corr',
             'parsed_prob_orig',
             'relative_pos',
             'span_size',
             'sumlm_ann_score',
             'sumlm_diff_score',
             'sumlm_orig_score',
             'total_Adjective_in_sent',
             'total_Adverb_in_sent',
             'total_Agreement_in_sent',
             'total_Conjunction_in_sent',
             'total_Determiner_in_sent',
             'total_Enhancement_in_sent',
             'total_Fluency_in_sent',
             'total_Morphology_in_sent',
             'total_Negation_in_sent',
             'total_Noun_in_sent',
             'total_OtherError_in_sent',
             'total_Preposition_in_sent',
             'total_Pronoun_in_sent',
             'total_Punctuation_in_sent',
             'total_Register_in_sent',
             'total_SentenceBoundary_in_sent',
             'total_SpellConfused_in_sent',
             'total_Spelling_in_sent',
             'total_Style_in_sent',
             'total_VerbSVA_in_sent',
             'total_VerbTense_in_sent',
             'total_Verb_in_sent',
             'total_WordOrder_in_sent',
             'total_anns_here',
             'total_anns_in_sent',
             'total_anns_other',
             'total_dublicates',
             'total_opc',
             'total_opc_here',
             'total_opc_other',
             'total_patterns',
             'total_patterns_here',
             'total_patterns_other',
             'total_unique_anns',
             'total_upc',
             'total_upc_here',
             'total_upc_other',
             'upc_prob']


def get_list_of_dependent_features():
    dependent_features = \
        ['n_Adjective_anns_are_here',
         'n_Adjective_anns_are_in_other_places',
         'n_Adverb_anns_are_here',
         'n_Adverb_anns_are_in_other_places',
         'n_Agreement_anns_are_here',
         'n_Agreement_anns_are_in_other_places',
         'n_Conjunction_anns_are_here',
         'n_Conjunction_anns_are_in_other_places',
         'n_Determiner_anns_are_here',
         'n_Determiner_anns_are_in_other_places',
         'n_Enhancement_anns_are_here',
         'n_Enhancement_anns_are_in_other_places',
         'n_Fluency_anns_are_here',
         'n_Fluency_anns_are_in_other_places',
         'n_Morphology_anns_are_here',
         'n_Morphology_anns_are_in_other_places',
         'n_Negation_anns_are_here',
         'n_Negation_anns_are_in_other_places',
         'n_Noun_anns_are_here',
         'n_Noun_anns_are_in_other_places',
         'n_OtherError_anns_are_here',
         'n_OtherError_anns_are_in_other_places',
         'n_Preposition_anns_are_here',
         'n_Preposition_anns_are_in_other_places',
         'n_Pronoun_anns_are_here',
         'n_Pronoun_anns_are_in_other_places',
         'n_Punctuation_anns_are_here',
         'n_Punctuation_anns_are_in_other_places',
         'n_Register_anns_are_here',
         'n_Register_anns_are_in_other_places',
         'n_SentenceBoundary_anns_are_here',
         'n_SentenceBoundary_anns_are_in_other_places',
         'n_SpellConfused_anns_are_here',
         'n_SpellConfused_anns_are_in_other_places',
         'n_Spelling_anns_are_here',
         'n_Spelling_anns_are_in_other_places',
         'n_Style_anns_are_here',
         'n_Style_anns_are_in_other_places',
         'n_VerbSVA_anns_are_here',
         'n_VerbSVA_anns_are_in_other_places',
         'n_VerbTense_anns_are_here',
         'n_VerbTense_anns_are_in_other_places',
         'n_Verb_anns_are_here',
         'n_Verb_anns_are_in_other_places',
         'n_WordOrder_anns_are_here',
         'n_WordOrder_anns_are_in_other_places',
         'total_Adjective_in_sent',
         'total_Adverb_in_sent',
         'total_Agreement_in_sent',
         'total_Conjunction_in_sent',
         'total_Determiner_in_sent',
         'total_Enhancement_in_sent',
         'total_Fluency_in_sent',
         'total_Morphology_in_sent',
         'total_Negation_in_sent',
         'total_Noun_in_sent',
         'total_OtherError_in_sent',
         'total_Preposition_in_sent',
         'total_Pronoun_in_sent',
         'total_Punctuation_in_sent',
         'total_Register_in_sent',
         'total_SentenceBoundary_in_sent',
         'total_SpellConfused_in_sent',
         'total_Spelling_in_sent',
         'total_Style_in_sent',
         'total_VerbSVA_in_sent',
         'total_VerbTense_in_sent',
         'total_Verb_in_sent',
         'total_WordOrder_in_sent',
         'total_anns_here',
         'total_anns_in_sent',
         'total_anns_other',
         'total_dublicates',
         'total_opc',
         'total_opc_here',
         'total_opc_other',
         'total_patterns',
         'total_patterns_here',
         'total_patterns_other',
         'total_unique_anns',
         'total_upc',
         'total_upc_here',
         'total_upc_other']
    return dependent_features


def get_all_indep_features():
    indep_features = ['Adjective_is_error_type_of_this_ann',
                      'Adverb_is_error_type_of_this_ann',
                      'Agreement_is_error_type_of_this_ann',
                      'Conjunction_is_error_type_of_this_ann',
                      'Determiner_is_error_type_of_this_ann',
                      'Enhancement_is_error_type_of_this_ann',
                      'Fluency_is_error_type_of_this_ann',
                      'Morphology_is_error_type_of_this_ann',
                      'Negation_is_error_type_of_this_ann',
                      'Noun_is_error_type_of_this_ann',
                      'OtherError_is_error_type_of_this_ann',
                      'Preposition_is_error_type_of_this_ann',
                      'Pronoun_is_error_type_of_this_ann',
                      'Punctuation_is_error_type_of_this_ann',
                      'Register_is_error_type_of_this_ann',
                      'SentenceBoundary_is_error_type_of_this_ann',
                      'SpellConfused_is_error_type_of_this_ann',
                      'Spelling_is_error_type_of_this_ann',
                      'Style_is_error_type_of_this_ann',
                      'VerbSVA_is_error_type_of_this_ann',
                      'VerbTense_is_error_type_of_this_ann',
                      'Verb_is_error_type_of_this_ann',
                      'WordOrder_is_error_type_of_this_ann',
                      'is_confused',
                      'is_delete',
                      'is_insert',
                      'is_opc',
                      'is_patterns',
                      'is_upc',
                      'kenlm_ann_score',
                      'kenlm_ann_score_normalized',
                      'kenlm_diff_score',
                      'kenlm_diff_score_normalized',
                      'kenlm_orig_score',
                      'kenlm_orig_score_normalized',
                      'len_tokens',
                      'parsed_diff',
                      'parsed_prob_corr',
                      'parsed_prob_orig',
                      'relative_pos',
                      'span_size',
                      'sumlm_ann_score',
                      'sumlm_ann_score_normalized',
                      'sumlm_diff_score',
                      'sumlm_diff_score_normalized',
                      'sumlm_orig_score',
                      'sumlm_orig_score_normalized',
                      'upc_prob']
    return indep_features


def add_new_features(record):
    fd = {}
    # kenlm rescore
    fd['kenlm_orig_score_normalized'] = \
        float(record['kenlm_orig_score'])/int(record['len_tokens'])
    fd['kenlm_ann_score_normalized'] = \
        float(record['kenlm_ann_score'])/int(record['len_tokens'])
    fd['kenlm_diff_score_normalized'] = fd['kenlm_ann_score_normalized'] - \
                                        fd['kenlm_orig_score_normalized']
    # sunlm rescore
    fd['sumlm_orig_score_normalized'] = \
        float(record['sumlm_orig_score'])/int(record['len_tokens'])
    fd['sumlm_ann_score_normalized'] = \
        float(record['kenlm_ann_score'])/int(record['len_tokens'])
    fd['sumlm_diff_score_normalized'] = fd['sumlm_ann_score_normalized'] - \
                                        fd['sumlm_orig_score_normalized']
    return fd


def process_record(record, features_names):
    fd = add_new_features(record)
    features_names = sorted(features_names + list(fd.keys()))
    record = {**record, **fd}
    features = []
    ordered_feature_names = []
    for feature in features_names:
        features.append(float(record[feature]))
        ordered_feature_names.append(feature)
    label = float(record.get('label', 0))
    # label = int(label > 0.5)
    return features, label, ordered_feature_names


def get_x_and_y(all_records, skip_dependent=False):
    features = []
    labels = []
    restricted_metanames = ['label', 'pname', 'error_type', 'system_type',
                            'max_prob', 'clf_score']
    if skip_dependent:
        dependent_features = get_list_of_dependent_features()
        restricted_metanames = restricted_metanames + dependent_features
    features_names = [x for x in sorted(all_records[0].keys())
                      if x not in restricted_metanames]
    for record in all_records:
        try:
            tmp_x, label, ordered_features_names = \
                process_record(record, features_names)
            features.append(tmp_x[:])
            labels.append(label)
        except Exception as e:
            print(record)
            continue
    features = np.array(features)
    labels = np.array(labels)
    return features, labels, ordered_features_names


def get_trees_param_dict():
    param_dict = {'max_depth': [2, 3, 4, 5, 6, 7],
                  'max_features': ['log2', 'sqrt', 0.05, 0.1, 0.2],
                  'n_estimators': [x for x in range(50, 1000)],
                  'random_state': [None, 42,],
                  'min_samples_leaf': [0.001, 0.01, 0.02, 0.05, 0.005]}
    return param_dict

def get_clf_list():
    clfs = [# LogisticRegression(penalty='l2'),
            # LogisticRegression(penalty='l1'),
            # RandomForestClassifier(n_estimators=10),
            # RandomForestClassifier(n_estimators=20),
            # RandomForestClassifier(n_estimators=33),
            # LGBMClassifier(n_estimators=33),
            # GradientBoostingClassifier(n_estimators=33),
            # RandomForestClassifier(n_estimators=50),
            RandomForestClassifier(),
            # LGBMClassifier(n_estimators=100),
            GradientBoostingClassifier(),]
            # ExtraTreesClassifier()]
            # RandomForestClassifier(n_estimators=200)]
    return clfs


def get_clf_pred_probs(clf, X_data):
    if "RandomForestRegressor" in str(clf):
        pred_probs = [[x] * 2 for x in clf.predict(X_data)]
    else:
        try:
            pred_probs = clf.predict_proba(X_data)
        except Exception:
            pred_probs = [[x] * 2 for x in clf.predict(X_data)]
    return pred_probs


def get_kenlm_preds(x_data, y_test, feature_names):
    pos = -1
    feature_size = x_data.shape[1]
    for i, f_name in enumerate(feature_names):
        if f_name == "kenlm_diff_score":
            pos = i
            break
    if pos > 0 and pos < feature_size:
        kenlm_preds = [int(x[pos] > 0) for x in x_data]
        acc_k = sum([int(p == y) for
                     p, y in zip(kenlm_preds, y_test)]) / len(y_test)
        print(f"Kenlm baseline is {acc_k}")


def calculate_and_print_metrics(y_test, pred_probs, threshold=0.5):
    y_test = [int(y > 0.5) for y in y_test]
    preds = [int(x[1] > threshold) for x in pred_probs]
    acc = sum([int(p==y) for p,y in zip(preds, y_test)])/len(y_test)
    # precision before clf
    tp_old = sum([int(y == 1) for p, y in zip(preds, y_test)])
    fp_old = sum([int(y == 0) for p, y in zip(preds, y_test)])
    pr_old = tp_old/(tp_old + fp_old)
    print(f"Before clf TP {tp_old}; FP {fp_old}; PRECISION "
          f"{pr_old}")
    # precision
    tp = sum([int(p == 1 and y == 1) for p, y in zip(preds, y_test)])
    fp = sum([int(p == 1 and y == 0) for p, y in zip(preds, y_test)])
    pr = tp/(tp + fp)
    print(f"After clf TP {tp}; FP {fp}; PRECISION {pr}")
    if tp_old:
        print(f"Recall become relatively lower on {round(1 - (tp/tp_old), 3)*100}%")
    print(f"FP rate become relatively lower on {round(1 - (fp/fp_old), 3)*100}%")
    if fp_old:
        print(f"Precision become absolute higher on {round(pr - pr_old, 3)*100}%")
    print(f"Test accuracy is equal {acc}")
    print(f"All 1 baseline is {sum(y_test)/len(y_test)}")
    print(f"All 0 baseline is {1 - sum(y_test)/len(y_test)}")
    return acc


def print_feature_weights(clf, features_names, top_n=10):
    try:
        weights = clf.coef_[0]
    except:
        weights = clf.feature_importances_
    coefs = [[abs(x), x, y] for x, y in zip(weights, features_names)]
    for i, [ab, w, f_name] in enumerate(reversed(sorted(coefs))):
        # print stats for top10 features only
        if i < top_n:
            print(f_name, round(w, 4))


def validate_model(clf, X_test, y_test, features_names):
    print(clf)
    pred_probs = get_clf_pred_probs(clf, X_test)
    acc = calculate_and_print_metrics(y_test, pred_probs)
    get_kenlm_preds(X_test, y_test, features_names)
    print_feature_weights(clf, features_names)
    return acc


def get_clf_scores(clf, scaler, selector, ann_tokens, conflict_anns,
                   other_anns_list):
    scores = []
    for ann in conflict_anns:
        features = get_features(ann_tokens, ann, conflict_anns, other_anns_list)
        features, labels, _ = get_x_and_y([features])
        features_norm = scaler.transform(features)
        features_selected = selector.transform(features_norm)
        pred_probs = get_clf_pred_probs(clf, features_selected)
        scores.append(pred_probs[0][1])
    return scores


def balance_by_error_types(features, labels, error_types):
    unique_error_types = sorted(list(set(error_types)))
    all_features, all_labels = [], []
    for et in unique_error_types:
        labels_selected = []
        features_selected = []
        for i in range(len(error_types)):
            if et == error_types[i]:
                labels_selected.append(labels[i])
                features_selected.append(features[i])
        balance_et = sum(labels_selected)/ len(labels_selected)
        print(f"Statistic for {et}. Total records {len(labels_selected)}. "
              f"Balance {balance_et}")
        balanced_features, balanced_labeles = balance_classes(features_selected,
                                                              labels_selected)
        all_features.extend(balanced_features)
        all_labels.extend(balanced_labeles)
    # shuffle both lists
    tmp = list(zip(all_features, all_labels))
    shuffle(tmp)
    all_features, all_labels = zip(*tmp)
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    print(f"Total size of records {len(all_features)}")
    return all_features, all_labels


def balance_classes(features, labels):
    start_balance = sum(labels)/len(labels)
    new_features, new_labels = [], []
    for x, y in zip(features, labels):
        rnd = random()
        if start_balance > 0.5:
            if y == 0 or rnd < (1 - start_balance)/start_balance:
                new_features.append(x)
                new_labels.append(y)
            else:
                continue
        else:
            if y == 1 or rnd < start_balance/(1 - start_balance):
                new_features.append(x)
                new_labels.append(y)
            else:
                continue
    new_balance = sum(new_labels)/len(new_labels)
    print(f"After rebalancing balance is {new_balance}. "
          f"Total records {len(new_features)}")
    new_features = np.array(new_features)
    new_labels = np.array(new_labels)
    return new_features, new_labels


def report_cv_results(results, n_top=5):
    # Utility function to report best scores
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def train_clf(train_file, model_file, skip_dependent_features=False,
              balance=False, error_types=[], n_jobs=16, n_iter=30, n_cv=3):
    print('Start loading data')
    features, labels, features_names, error_types_list, _ = \
        get_processed_records(train_file, error_types, store_sents=False,
                              skip_dep_features=skip_dependent_features)
    print(f"There are {len(features)} features")

    scaler = StandardScaler()
    scaler.fit(features)
    features_norm = scaler.transform(features)
    '''
    print("Features min/max before/after scaling.")
    for i in range(len(features[0])):
        l1 = [x[i] for x in features]
        l2 = [x[i] for x in features_norm]
        print(min(l1), max(l1))
        print(min(l2), max(l2))
        print('\n')
    print(scaler.mean_)
    '''
    if balance:
        b_features, b_labels = balance_by_error_types(features_norm, labels,
                                                      error_types_list)
    else:
        b_features, b_labels = features_norm, labels
    features_norm, labels = [], []
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(b_features,
                                                                b_labels,
                                                                test_size=0.1,
                                                                random_state=42)
    b_features, b_labels = [], []
    best_acc = 0
    stats_pairs = []
    param_dict = get_trees_param_dict()
    for clf in get_clf_list():
        y_test = np.array([int(x > 0.5) for x in y_test_raw])
        y_train = np.array([int(x > 0.5) for x in y_train_raw])
        # clf_selector = deepcopy(clf)
        print(f"Start fitting selector {str(clf)}")
        random_search = RandomizedSearchCV(clf, param_dict, n_iter=n_iter,
                                           cv=n_cv, verbose=2, n_jobs=n_jobs)
        random_search.fit(X_train, y_train)
        report_cv_results(random_search.cv_results_, n_top=min(n_iter, 5))
        clf_selector = random_search.best_estimator_
        # clf_selector.fit(X_train, y_train)
        test_acc_before_selecting = validate_model(clf_selector, X_test,
                                                   y_test, features_names)
        # add feature selecting
        selector = SelectFromModel(clf_selector, prefit=True)
        shape_before = X_test.shape[1]
        X_test_pruned = selector.transform(X_test)
        X_train_pruned = selector.transform(X_train)
        selected_idx = \
            list(selector.transform(np.array([list(range(shape_before))]))[0])
        selected_features = []
        for i, f_name in enumerate(features_names):
            if i in selected_idx:
                selected_features.append(f_name)
        shape_after = X_test_pruned.shape[1]
        print(f"Start fitting model {str(clf)}")
        random_search = RandomizedSearchCV(clf, param_dict, n_iter=n_iter,
                                           cv=n_cv, verbose=2, n_jobs=n_jobs)
        random_search.fit(X_train_pruned, y_train)
        report_cv_results(random_search.cv_results_, n_top=min(n_iter, 5))
        clf_model = random_search.best_estimator_
        test_acc = validate_model(clf_model, X_test_pruned, y_test,
                                  selected_features)
        print(
            f"Acc before pruning {test_acc_before_selecting} with {shape_before} features")
        print(f"Acc after pruning {test_acc} with {shape_after} features")
        stats_pairs.append([test_acc, str(random_search.best_estimator_)])
        model_name = type(clf_model).__name__
        model_iter_name = model_file.replace(".pkl", f"_{model_name}.pkl")
        dump_model(model_iter_name, clf_model, scaler, selector)
        if test_acc > best_acc:
            best_acc = test_acc
            best_clf = clf_model
            best_selector = selector
    print("Overall stats")
    for acc, model in stats_pairs:
        print(model)
        print(f"Test acc was {acc}")
        print('\n\n')
    # clean up memory
    X_test, X_test_pruned, X_train, X_train_pruned = [], [], [], []
    y_train_raw, y_train, y_test_raw, y_test = [], [], [], []

    print(str(best_clf), best_acc)
    dump_model(model_file, best_clf, scaler, best_selector)


def dump_model(model_name, clf, scaler, selector):
    pickle.dump(clf, open(model_name.replace(".pkl", "_model.pkl"), 'wb'))
    pickle.dump(scaler, open(model_name.replace(".pkl", "_scaler.pkl"), 'wb'))
    pickle.dump(selector, open(model_name.replace(".pkl", "_selector.pkl"), 'wb'))


def load_models(model_name):
    clf_name = model_name.replace(".pkl", "_model.pkl")
    clf = pickle.load(open(clf_name, 'rb'))
    scaler_name = model_name.replace(".pkl", "_scaler.pkl")
    scaler = pickle.load(open(scaler_name, 'rb'))
    selector_name = model_name.replace(".pkl", "_selector.pkl")
    selector = pickle.load(open(selector_name, 'rb'))
    return clf, scaler, selector


def process_line_with_clf(comb_data):
    clf, scaler, selector, skip_deps, line = comb_data
    ann_tokens = AnnotatedTokens(AnnotatedText(line))
    for ann in ann_tokens.iter_annotations():
        features, labels, ordered_features_names = get_x_and_y([ann.meta],
                                                               skip_deps)
        features_norm = scaler.transform(features)
        features_selected = selector.transform(features_norm)
        pred_probs = get_clf_pred_probs(clf, features_selected)
        score = pred_probs[0][1]
        ann.meta['clf_score'] = score
    return ann_tokens.get_annotated_text()


def add_clf_scores_to_file(model_file, eval_file, error_types, skip_deps=False):
    print("Loading models")
    clf, scaler, selector = load_models(model_file)
    print("Loading data records")
    records, all_ann_sents = get_all_records(eval_file, error_types=error_types)
    print(f"{len(records)} records and {len(all_ann_sents)} "
          f"sentences were loaded")
    print("Doing feature preprocessing")
    features, labels, features_names = get_x_and_y(records, skip_deps)
    labels = np.array([int(x > 0.5) for x in labels])
    features_norm = scaler.transform(features)
    features_selected = selector.transform(features_norm)
    print("Doing prediction")
    scores = get_clf_pred_probs(clf, features_selected)
    print("Adding scores to sentences")
    cnt = 0
    label_stats =[]
    for ann_sent in all_ann_sents:
        for ann in ann_sent.iter_annotations():
            ann.meta['clf_score'] = scores[cnt][1]
            label_stats.append(float(ann.meta['label']))
            cnt += 1
    assert cnt == len(records)
    print("Calculate accuracy")
    acc = calculate_and_print_metrics(labels, scores)
    print("Saving results to the file")
    output = [x.get_annotated_text() for x in all_ann_sents]
    clf_name = os.path.basename(model_file).replace(".pkl", "")
    et = "None" if not error_types else " ".join(error_types)
    out_file = eval_file.replace(".txt", f"_scored_by_{clf_name}_on_{et}.txt")
    write_lines(out_file, output)


def main(args):
    if args.error_type is None:
        error_types = []
    else:
        error_types = args.error_type.split()
    if args.task == "train":
        train_clf(args.train_file, args.model_file,
                  args.skip_dependent_features, args.balance, error_types,
                  args.n_jobs, args.n_iter, args.n_cv)
    elif args.task == "test":
        add_clf_scores_to_file(args.model_file, args.eval_file, error_types,
                               args.skip_dependent_features)
    else:
        raise Exception(f"Unknown {args.task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file',
                        help='Path to the output model file')
    parser.add_argument('--train_file',
                        help='Path to the file with annotations with features')
    parser.add_argument('--eval_file',
                        help='Path to the file with annotations with features')
    parser.add_argument('--balance',
                        action='store_true',
                        help='Enable balancing classes',
                        default=False)
    parser.add_argument('--skip_dependent_features',
                        action='store_true',
                        help='If set then all dependent features '
                             'will be skipped.',
                        default=False)
    parser.add_argument('--error_type',
                        help='Set if you want to filter error only from '
                             'one error type.',
                        default=None)
    parser.add_argument('--task',
                        help='Specify what you want to do.',
                        choices=['train', 'test'],
                        default='train')
    parser.add_argument('--n_jobs',
                        help='Specify how many computations you want to do '
                             'in parallel.',
                        type=int,
                        default=16)
    parser.add_argument('--n_iter',
                        help='Specify how many iterations of random search '
                             'you want to do.',
                        type=int,
                        default=30)
    parser.add_argument('--n_cv',
                        help='Specify how many folds you will during'
                             'cross validation.',
                        type=int,
                        default=3)
    args = parser.parse_args()
    main(args)

#!/usr/bin/env python
"""
Script for combining submissions with AnnotatedTokens in different ways.
"""
import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from grampy.text import AnnotatedText, AnnotatedTokens, Annotation
from grampy.api import upc_check, opc_check, check
from grampy.lm import ngram_client

from scorer.classifier.get_features import split_annotations_on_disjoint_pairs, \
    fix_upc_output, fix_patterns_output, fix_opc_output,\
    collect_other_anns_list, get_normalized_error_type



def get_kenlm_scores(ann_tokens, ann_list, add_original=False, use_norm=False):
    """Score all annotations in the list using KenLM"""
    kenlm = ngram_client.KenLMClient()
    sent_list = []
    for ann in ann_list:
        sent = AnnotatedTokens(ann_tokens._tokens)
        sent.annotate(ann.start, ann.end, ann.suggestions)
        sent_list.append(sent.get_corrected_text())
    if add_original:
        sent_list.append(sent.get_original_text())
    scores = kenlm.ask_ngrams(sent_list)
    if use_norm:
        scores = [scores[i]/max(len(sent_list[i].split()), 1)
                  for i in range(len(scores))]
    return scores


def is_anns_different(ann_list):
    """Check if at least two annotations in the list are different"""
    first_ann = ann_list[0]
    for ann in ann_list[1:]:
        if ann.start != first_ann.start:
            return True
        if ann.end != first_ann.end:
            return True
        if ann.suggestions[0] != first_ann.suggestions[0]:
            return True
    return False


def apply_discards(source_list, priority_order_list):
    """Simple discard logic"""
    for n_priority in priority_order_list:
        if n_priority in source_list:
            return source_list.index(n_priority)
    raise Exception("Undefined system name")


def resolve_annotation_conflicts(ann_tokens, conflict_anns,
                                 strategy, n_options, discard_priorities):
    """Get the best available annotation from the given list of
    confict annotations """
    total_size = len(conflict_anns)
    if strategy == 'all-agree':
        if total_size < n_options or is_anns_different(conflict_anns):
            return None
        else:
            new_ann = conflict_anns[0]
            return new_ann
    if strategy == "kenlm" and total_size > 1:
        idx = 0
        if is_anns_different(conflict_anns):
            scores = get_kenlm_scores(ann_tokens, conflict_anns)
            idx = scores.index(max(scores))
        return conflict_anns[idx]
    if strategy.startswith("kenlm-orig"):
        use_norm = True if strategy.endswith("norm") else False
        scores = get_kenlm_scores(ann_tokens, conflict_anns, add_original=True,
                                  use_norm=use_norm)
        idx = scores.index(max(scores))
        if idx >= len(conflict_anns):
            return None
        else:
            return conflict_anns[idx]
    if strategy.startswith("clf"):
        threshold = float(strategy.split("-")[1])
        best_acc = threshold
        best_ann = None
        for ann in conflict_anns:
            clf_score = float(ann.meta['clf_score'])
            if clf_score > best_acc:
                best_acc = clf_score
                best_ann = ann
        if best_ann:
            return best_ann
        else:
            return None
    if total_size == 1:
        return conflict_anns[0]
    elif total_size == 2:
        ann1, ann2 = conflict_anns
        if strategy == "extend-suggestions":
            if ann1.start == ann2.start and ann1.end == ann2.end:
                suggestions = list(set(ann1.suggestions + ann2.suggestions))
                new_ann = Annotation(ann1.start, ann1.end, ann1.source_text,
                                     suggestions, meta={"system_type": "combined"})
            else:
                source_list = [ann1.meta['system_type'],
                               ann2.meta['system_type']]
                idx = apply_discards(source_list, discard_priorities)
                new_ann = conflict_anns[idx]
            return new_ann
        elif strategy in "priority-discard":
            source_list = [ann1.meta['system_type'], ann2.meta['system_type']]
            idx = apply_discards(source_list, discard_priorities)
            new_ann = conflict_anns[idx]
            return new_ann
        else:
            raise Exception("Unknown strategy")
    elif total_size > 2:
        # TODO: add better logic
        source_list = [conflict_anns[i].meta['system_type']
                       for i in range(total_size)]
        idx = apply_discards(source_list, discard_priorities)
        new_ann = conflict_anns[idx]
        return new_ann
    else:
        raise Exception("Logic is broken somewhere")


def combine_systems_output(sentence_list, strategy, discard_priorities):
    """Get one sentence which combined outputs of all systems"""
    all_anns = []
    tokens_list = []
    n_options = len(sentence_list)
    for sent in sentence_list:
        ann_tokens = get_ann_tokens(sent)
        tokens_list.append(ann_tokens._tokens)
        all_anns.extend([x for x in ann_tokens.iter_annotations()])
    paired_anns = split_annotations_on_disjoint_pairs(all_anns)
    tokens = tokens_list[0]
    ann_tokens = AnnotatedTokens(tokens)
    for i, comb_anns_list in enumerate(paired_anns):
        ann = resolve_annotation_conflicts(ann_tokens, comb_anns_list,
                                           strategy, n_options,
                                           discard_priorities)

        if ann and ann.suggestions and str(ann.suggestions[0]) != "NO_SUGGESTIONS":
            ann_tokens.annotate(ann.start, ann.end, ann.suggestions, ann.meta)
    return ann_tokens


def get_system_type_by_filename(filename):
    """Get name of the system assuming that it is used in filename."""
    base_file_name = os.path.basename(filename)
    if 'OPC' in base_file_name:
        return 'OPC'
    elif 'UPC' in base_file_name:
        return 'UPC'
    elif 'Patterns' in base_file_name:
        return 'Patterns'
    else:
        raise Exception("Incorrect filename")


def get_all_sentences(input_dir, discard_priorities, error_type=None):
    """Get list of all sentences from all systems"""
    filenames = [os.path.join(input_dir, x) for x in os.listdir(input_dir)
                 if x.endswith(".txt")]
    # gather all available systems
    systems_dict = {}
    for filename in filenames:
        system_type = get_system_type_by_filename(filename)
        if all(x not in system_type for x in discard_priorities):
            continue
        sents = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                ann_tokens = get_ann_tokens(line)
                for ann in ann_tokens.iter_annotations():
                    # remove some error types
                    if error_type is not None:
                        ann_error_type = get_normalized_error_type(ann)
                        if error_type != ann_error_type:
                            ann_tokens.remove(ann)
                            continue
                    ann.meta['system_type'] = system_type
                sents.append(ann_tokens.get_annotated_text(with_meta=True))

        systems_dict[system_type] = sents[:]
    # merge all systems to one list
    all_sentences = []
    for all_sents in zip(*list(systems_dict.values())):
        all_sentences.append(all_sents)
    return all_sentences


def get_ann_tokens(sent):
    """Remove annotation artifacts"""
    if isinstance(sent, AnnotatedText):
        ann_tokens = AnnotatedTokens(AnnotatedText(sent.get_annotated_text()))
    elif isinstance(sent, str):
        ann_tokens = AnnotatedTokens(AnnotatedText(sent.rstrip('\n')))
    else:
        raise Exception("Incorrect input for normalization!")
    return ann_tokens


def get_systems_output(sent):
    """Get dictionary with output of 3 systems"""
    sentence_list = []
    # get output for OPC
    ann_tokens = get_ann_tokens(fix_opc_output(opc_check(sent)))
    for ann in ann_tokens.iter_annortations():
        ann.meta['system_type'] = "OPC"
    sentence_list.append(ann_tokens)

    # get output for UPC
    upc_addr = 'upc-high-recall-server.phantasm.gnlp.io:8081'
    ann_tokens = get_ann_tokens(fix_upc_output(upc_check(sent, addr=upc_addr,
                                                         custom_server=True)))
    for ann in ann_tokens.iter_annortations():
        ann.meta['system_type'] = "UPC"
    sentence_list.append(ann_tokens)

    # get output for patterns
    ann_tokens = get_ann_tokens(fix_patterns_output(check(sent)[0]))
    for ann in ann_tokens.iter_annortations():
        ann.meta['system_type'] = "Patterns"
    sentence_list.append(ann_tokens)
    return sentence_list


def wrap_combine_systems_output(combined_record):
    sentence_list, args.task, discard_priorities = combined_record
    combined_sent = ""
    while not combined_sent:
        try:
            combined_sent = combine_systems_output(sentence_list, args.task,
                                                   discard_priorities)
        except Exception as e:
            print(e)
            print(sentence_list)
            print("Something went wrong. Sleep for 5 sec")
            sleep(5)
    return combined_sent


def main(args):
    # convert_to_corrected_sentences(args.input_file)
    discard_priorities = args.discard_order.split("-")
    all_sents = get_all_sentences(args.input_dir, discard_priorities,
                                  args.error_type)
    print("All sentences are loaded")
    combined_records = [[x, args.task, discard_priorities]
                        for x in all_sents]
    with ThreadPoolExecutor(args.n_threads) as pool:
        combined_sentences = pool.map(wrap_combine_systems_output,
                                      combined_records)
    combined_sentences = [x.get_annotated_text() for x in combined_sentences]
    out_file = args.out_file.replace(".txt",
                                     f"_{args.discard_order}_{args.task}.txt")
    with open(out_file, 'w') as f:
        f.write('\n'.join(combined_sentences) + '\n')
    print(len(combined_sentences))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir',
                        help='Path to the directory with files')
    parser.add_argument('out_file',
                        help='Path to the output file')
    parser.add_argument('--task',
                        help='Specify combination_strategy',
                        choices=['all-agree', 'extend-suggestions',
                                 'priority-discard', 'kenlm', 'kenlm-orig',
                                 'kenlm-orig-norm', 'clf-0.5', 'clf-0.0'],
                        default='kenlm-orig')
    parser.add_argument('--discard-order',
                        help='Specify the discard order',
                        default='Patterns')
    parser.add_argument('--error_type',
                        help='Set if you want to filter error only from '
                             'one error type.',
                        default=None)
    parser.add_argument('--n_threads',
                        type=int,
                        help='The number of threads.',
                        default=100)
    args = parser.parse_args()
    main(args)

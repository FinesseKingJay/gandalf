import argparse
import os
import re
from time import sleep
import pickle
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pyxdameraulevenshtein import damerau_levenshtein_distance

from grampy.text import AnnotatedText, AnnotatedTokens, Annotation, OnOverlap
from grampy.lm import ngram_client, SumLM
from grampy.api.capi import parse
from grampy.data.dictionaries import names, words
from grampy.api import check, opc_check, upc_check, ApiError
from scorer.helpers.utils import read_lines, write_lines
from scorer.helpers.error_types_bank import ErrorTypesBank


def sentence_processor_wrapper(sent_pair):
    sent, error_type, system_types = sent_pair
    response = []
    while not response:
        try:
            ann_sent, ann_dict = collect_all_features(sent, error_type, system_types)
            response = [ann_sent, ann_dict]
            return ann_sent, ann_dict
        except Exception as e:
            print(e)
            print(f"Something went wrong on {sent}")
            continue


def get_annotated_sent(sent):
    if isinstance(sent, str):
        return AnnotatedTokens(AnnotatedText(sent))
    elif isinstance(sent, AnnotatedText):
        return AnnotatedTokens(sent)
    elif isinstance(sent, AnnotatedTokens):
        return AnnotatedTokens
    else:
        raise Exception("Incorrect input")


def is_anns_intersect(ann1, ann2):
    """Check if two annotations intersect"""
    s1, e1 = ann1.start, ann1.end
    s2, e2 = ann2.start, ann2.end
    if s1 == s2 and e1 == e2:
        return True
    if s1 > s2 and s1 < e2:
        return True
    if s2 > s1 and s2 < e1:
        return True
    if e1 > s1 and e2 > s2 and e1 <= e2 and s1 >= s2:
        return True
    if e1 > s1 and e2 > s2 and e2 <= e1 and s2 >= s1:
        return True
    return False


def is_ann_intersect_with_any(ann, ann_list):
    return any(is_anns_intersect(ann, x) for x in ann_list)


def split_annotations_on_disjoint_pairs(list_of_anns):
    """Split all annotations in a list on pairs which is not intersect
    with each other"""
    total_size = len(list_of_anns)
    if not list_of_anns:
        return []
    anns_list = [(list_of_anns[i].start, list_of_anns[i].end, i, list_of_anns[i])
                 for i in range(total_size)]
    anns_list = sorted(anns_list)
    anns_list = [x[3] for x in anns_list]
    paired_anns = []
    while anns_list:
        ann1 = anns_list[0]
        anns_list = anns_list[1:]
        records_which_intersect = [ann1]
        idx_to_remove = []
        for i, ann2 in enumerate(anns_list):
            if is_ann_intersect_with_any(ann2, records_which_intersect):
                records_which_intersect.append(ann2)
                idx_to_remove.append(i)
        if len(records_which_intersect) > 1:
            anns_list = [anns_list[i] for i in range(len(anns_list))
                         if i not in idx_to_remove]
        paired_anns.append(records_which_intersect[:])
    return paired_anns


def get_kenlm_scores(sub, ann):
    fd = {}
    kenlm = ngram_client.KenLMClient()
    orig_sent = sub.get_original_text()
    new_ann_sent = AnnotatedTokens(AnnotatedText(orig_sent))
    new_ann_sent.annotate(ann.start, ann.end, ann.suggestions)
    corr_sent = new_ann_sent.get_corrected_text()
    orig_prob, corr_prob = kenlm.ask_ngrams([orig_sent, corr_sent])
    orig_prob = max(orig_prob, -5000)
    corr_prob = max(corr_prob, -5000)
    fd['kenlm_orig_score'] = orig_prob
    fd['kenlm_ann_score'] = corr_prob
    fd['kenlm_diff_score'] = corr_prob - orig_prob
    return fd


def get_sumlm_scores(sub, ann):
    fd = {}
    sumlm = SumLM(min_level=3)
    orig_sent = sub.get_original_text()
    new_ann_sent = AnnotatedTokens(AnnotatedText(orig_sent))
    new_ann_sent.annotate(ann.start, ann.end, ann.suggestions)
    corr_sent = new_ann_sent.get_corrected_text()
    orig_score, corr_score = sumlm.score_many([orig_sent, corr_sent])
    fd['sumlm_orig_score'] = orig_score
    fd['sumlm_ann_score'] = corr_score
    fd['sumlm_diff_score'] = corr_score/orig_score if orig_score else 0
    return fd


def get_capi_scores(sub, ann):
    fd = {}
    orig_sent = sub.get_original_text()
    new_ann_sent = AnnotatedTokens(AnnotatedText(orig_sent))
    new_ann_sent.annotate(ann.start, ann.end, ann.suggestions)
    corr_sent = new_ann_sent.get_corrected_text()
    parsed_orig = parse(orig_sent)
    parsed_corr = parse(corr_sent)
    fd['parsed_prob_orig'] = get_parse_prob(parsed_orig)
    fd['parsed_prob_corr'] = get_parse_prob(parsed_corr)
    fd['parsed_diff'] = fd['parsed_prob_corr'] - fd['parsed_prob_orig']
    return fd


def get_parse_prob(parsed_json):
    if 'texts' in parsed_json:
        texts = parsed_json['texts']
        if texts and 'paragraphs' in texts[0]:
            paragraphs = texts[0]['paragraphs']
            if paragraphs and 'sentences' in paragraphs[0]:
                sentences = paragraphs[0]['sentences']
                if sentences and 'parseProb' in sentences[0]:
                    return sentences[0].get('parseProb', 0)
    return 0



def get_error_type_features(ann, all_anns, all_other_anns):
    error_types = ErrorTypesBank().get_error_types_list('Patterns22') + ['OtherError']
    ann_error_type = get_normalized_error_type(ann)
    all_anns = [get_normalized_error_type(x) for x in all_anns]
    all_other_anns = [get_normalized_error_type(x) for x in all_other_anns]
    fd = {}
    for error_type in error_types:
        fd[f'{error_type}_is_error_type_of_this_ann'] = \
            int(error_type == ann_error_type)
        fd[f'n_{error_type}_anns_are_here'] = \
            sum([1 for x in all_anns if x == error_type])
        fd[f'n_{error_type}_anns_are_in_other_places'] = \
            sum([1 for x in all_other_anns if x == error_type])
        fd[f'total_{error_type}_in_sent'] = \
            fd[f'n_{error_type}_anns_are_here'] + \
            fd[f'n_{error_type}_anns_are_in_other_places']
    # fd['total_anns_in_sent'] = len(all_anns) + len(all_other_anns)
    return fd

def get_ann_type_features(sub, ann, all_anns, all_other_anns):
    fd = {}
    tokens = sub._tokens
    len_source = len(ann.source_text.split())
    len_corr = len(ann.suggestions[0].split()) if ann.suggestions else 0
    fd['is_insert'] = 1 if len_source < len_corr else 0
    fd['is_delete'] = 1 if len_source > len_corr else 0
    fd['is_confused'] = 1 if len_source == len_corr else 0
    fd['span_size'] = ann.end - ann.start
    fd['is_opc'] = 1 if 'OPC' == ann.meta['system_type'] else 0
    fd['is_upc'] = 1 if 'UPC' == ann.meta['system_type'] else 0
    fd['is_patterns'] = 1 if 'Patterns' == ann.meta['system_type'] else 0
    fd['len_tokens'] = len(sub._tokens)
    fd['relative_pos'] = ann.start/len(sub._tokens)
    fd['total_patterns_here'] = sum([1 for x in all_anns if x.meta['system_type'] == "Patterns"])
    fd['total_opc_here'] = sum([1 for x in all_anns if x.meta['system_type'] == "OPC"])
    fd['total_upc_here'] = sum([1 for x in all_anns if x.meta['system_type'] == "UPC"])
    fd['total_anns_here'] = fd['total_patterns_here'] + fd['total_opc_here'] + fd['total_upc_here']
    fd['total_patterns_other'] = \
        sum([1 for x in all_other_anns if x.meta['system_type'] == "Patterns"])
    fd['total_opc_other'] = \
        sum([1 for x in all_other_anns if x.meta['system_type'] == "OPC"])
    fd['total_upc_other'] = \
        sum([1 for x in all_other_anns if x.meta['system_type'] == "UPC"])
    fd['total_anns_other'] = fd['total_patterns_other'] + \
                             fd['total_opc_other'] + fd['total_upc_other']
    fd['total_patterns'] = fd['total_patterns_here'] + fd['total_patterns_other']
    fd['total_opc'] = fd['total_opc_here'] + fd['total_opc_other']
    fd['total_upc'] = fd['total_upc_here'] + fd['total_upc_other']
    fd['total_anns_in_sent'] = fd['total_patterns'] + fd['total_opc'] + \
                               fd['total_upc']

    fd['total_unique_anns'] = len(get_unique_anns(tokens, all_anns))
    fd['total_dublicates'] = fd['total_anns_here'] - fd['total_unique_anns']
    fd['upc_prob'] = ann.meta.get('max_prob', 0)
    return fd


def get_features(sub, ann, all_anns, all_other_anns):
    fd_kenlm = get_kenlm_scores(sub, ann)
    fd_sumlm = get_sumlm_scores(sub, ann)
    fd_capi = get_capi_scores(sub, ann)
    fd_ann = get_ann_type_features(sub, ann, all_anns, all_other_anns)
    fd_error_types = get_error_type_features(ann, all_anns, all_other_anns)
    features = {**fd_kenlm, **fd_sumlm, **fd_capi, **fd_ann, **fd_error_types}
    return features


def is_anns_the_same(tokens, gold_ann, sub_ann, sensitive=False, verbose=False):
    # very simple logic for comparison annotations
    gold_sent = AnnotatedTokens(tokens)
    gold_sent.annotate(gold_ann.start, gold_ann.end, gold_ann.suggestions)
    gold_corr = gold_sent.get_corrected_text()
    orig_sent = gold_sent.get_original_text()
    corrs = []
    for sugg in sub_ann.suggestions:
        sub_sent = AnnotatedTokens(tokens)
        sub_sent.annotate(sub_ann.start, sub_ann.end, sugg)
        corrs.append(sub_sent.get_corrected_text())
    if not sensitive:
        orig_sent = orig_sent.lower().replace(" ", "")
        gold_corr = gold_corr.lower().replace(" ", "")
        corrs = [x.lower().replace(" ", "") for x in corrs]
    is_same = 0
    for i, corr in enumerate(corrs):
        if gold_corr == corr:
            is_same = 1/(1 + i)
            break
    # if not a perfect match then use some heuristics
    if is_anns_intersect(gold_ann, sub_ann) and not is_same and corrs:
        gold_to_orig = damerau_levenshtein_distance(gold_corr, orig_sent)
        first_corr = corrs[0]
        sub_to_gold = damerau_levenshtein_distance(gold_corr, first_corr)
        sub_to_orig = damerau_levenshtein_distance(first_corr, orig_sent)
        # TODO: check this ED logic
        if sub_to_gold < gold_to_orig:
                is_same = float((gold_to_orig - sub_to_gold)/gold_to_orig)
        if verbose:
            gold_stats = [gold_ann.start, gold_ann.end, gold_ann.source_text,
                          gold_ann.suggestions]
            sub_stats = [sub_ann.start, sub_ann.end, sub_ann.source_text,
                         sub_ann.suggestions]
            print(
                f"The anns are different.\nGold {gold_stats}\nSub {sub_stats}")
            print(f"ED values are GOLD_TO_ORIG {gold_to_orig}; "
                  f"SUB_TO_GOLD {sub_to_gold}; "
                  f"SUB_TO_ORIG {sub_to_orig}; is_same_value {is_same}")
            print(f"GOLD corr {gold_corr}")
            print(f"Orig sent {orig_sent}")
            print(f"First corr {first_corr}\n")
    return is_same


def get_unique_anns(tokens, anns_list):
    unique_list = []
    for ann in anns_list:
        if ann not in unique_list:
            ann_is_unique = True
            for un_ann in unique_list:
                if is_anns_the_same(tokens, un_ann, ann):
                    ann_is_unique = False
            if ann_is_unique:
                unique_list.append(ann)
    return unique_list


def get_label(gold_sent, sub_ann):
    tokens = gold_sent._tokens
    for gold_ann in gold_sent.iter_annotations():
        is_same = is_anns_the_same(tokens, gold_ann, sub_ann,
                                   sensitive=False, verbose=True)
        if is_same:
            return is_same
    return 0


def modify_ann(sub, gold, ann, all_anns, all_other_anns):
    features_dict = get_features(sub, ann, all_anns, all_other_anns)
    label = get_label(gold, ann)
    for key, value in features_dict.items():
        ann.meta[key] = value
    ann.meta['label'] = label
    return ann


def process_sentence_pair(sent_pair):
    gold_sent, sub_sent, system_name = sent_pair
    gold_ann_sent = get_annotated_sent(gold_sent)
    sub_ann_sent = get_annotated_sent(sub_sent)
    for ann in sub_ann_sent.iter_annotations():
        modify_ann(sub_ann_sent, gold_ann_sent, ann, system_name)
    return sub_ann_sent


def get_normalized_error_type(ann):
    eb = ErrorTypesBank()
    ann_error_type = ann.meta.get('error_type', 'OtherError')
    pname = ann.meta.get('pname', 'OtherError')
    system_type = ann.meta.get('system_type')
    if system_type == "Patterns":
        norm_error_type = eb.pname_to_patterns22(pname)
    elif system_type == "UPC":
        norm_error_type = eb.upc5_to_patterns22(ann_error_type)
    elif system_type == "OPC":
        norm_error_type = eb.opc_to_patterns22(ann_error_type)
    else:
        print(f"Something went wrong with {ann}")
        norm_error_type = "OtherError"
    return norm_error_type


def smart_capitalize(word):
    words = word.split()
    if not words:
        return word
    words = [words[0].capitalize()] + words[1:]
    return " ".join(words)


def fix_patterns_output(ann_sent):
    sent = ann_sent.get_original_text()
    anns_for_removing = []
    anns_for_adding = []
    for ann in ann_sent.iter_annotations():
        ann.meta['system_type'] = "Patterns"
        error_type = get_normalized_error_type(ann)
        if error_type in ["Punctuation", "SentenceBoundary"]:
            # insertion case
            if ann.source_text == "":
                suggestions = [" " + x + " " for x in ann.suggestions]
                anns_for_removing.append(ann)
                new_ann = Annotation(ann.start, ann.end, ann.source_text,
                                     suggestions, ann.meta)
                anns_for_adding.append(new_ann)
            # insertion with a space case
            elif ann.source_text.strip() == "" and (ann.end - ann.start) == 1:
                suggestions = [" " + x.strip() + " " for x in ann.suggestions]
                anns_for_removing.append(ann)
                new_ann = Annotation(ann.end, ann.end, ann.source_text,
                                     suggestions, ann.meta)
                anns_for_adding.append(new_ann)
            # deletion case
            elif len(ann.suggestions) == 1 and not ann.suggestions[0] \
                    and len(ann.source_text.strip()) < ann.end - ann.start:
                anns_for_removing.append(ann)
                new_ann = Annotation(ann.start,
                                     ann.start + len(ann.source_text.strip()),
                                     ann.source_text, ann.suggestions, ann.meta)
                anns_for_adding.append(new_ann)
            # confused with extra space case
            elif ann.source_text.startswith(" ") \
                    and sent[ann.start:ann.start + 1] == " " \
                    and sent[max(ann.start - 1, 0):ann.start] != " " \
                    and (ann.end - ann.start) > 1:
                anns_for_removing.append(ann)
                new_ann = Annotation(ann.start + 1, ann.end, ann.source_text,
                                     ann.suggestions, ann.meta)
                anns_for_adding.append(new_ann)
            # annotations without suggestions case
            elif not ann.suggestions:
                anns_for_removing.append(ann)
            else:
                pass
    # remove if any
    if anns_for_removing:
        for ann in anns_for_removing:
            try:
                ann_sent.remove(ann)
            except ValueError:
                continue
    # add if any
    if anns_for_adding:
        for ann in anns_for_adding:
            ann_sent.annotate(ann.start, ann.end, ann.suggestions,
                              meta=ann.meta, on_overlap=OnOverlap.OVERRIDE)
    # add regex fix
    sent = ann_sent.get_annotated_text()
    ann_sent = AnnotatedText(punct_alert_fixer(sent))
    return ann_sent


def punct_alert_fixer(sentence):
    # {" .=>."} ==> {" .=>. "}
    sentence = re.sub(r'\{(\S+) (\S+)=>\2\1\}', r'{\1 \2=>\2 \1}', sentence)
    # {1th ?=>1st?} ==> {1th=>1st} ?
    sentence = re.sub(r'\{(\S+(th|st|rd|nd)) ([^A-z])=>(\S+(th|st|rd|nd))\3\}',
                      r'{\1=>\4} \3', sentence)
    # {Tom=>, Tom,} ==> {Tom=>, Tom ,}
    # {for example=>, for example,} ==> {for example=>, for example ,}
    sentence = re.sub(r'\{([^=]*?)=>([^A-z]) \1(\2)\}', r'{\1=>\3 \1 \3}',
                      sentence)
    # {ie ,=>i.e.,} ==> {ie ,=>i.e. ,}
    sentence = re.sub(r'([A-z]|[A-z]\.)(,|:|;|!|\?|\)|\.\.\.)', r'\1 \2',
                      sentence)
    # {etc )=>etc. )} ==> {etc=>etc.} )
    sentence = re.sub(r'\{(\S+) (\)|;|:|,)=>(\S+) (\2)\}', r'{\1=>\3} \2',
                      sentence)
    #print(sentence)
    return sentence


def fix_upc_output(ann_sent):
    ann_tokens = AnnotatedTokens(ann_sent)
    new_ann_tokens = AnnotatedTokens(ann_tokens._tokens)
    is_all_upper = all(x == x.upper() for x in ann_tokens._tokens)
    for ann in ann_tokens.iter_annotations():
        suggestions = ann.suggestions
        if not suggestions:
            ann_sent.remove(ann)
            continue
        first_suggestion = suggestions[0]
        if is_all_upper:
            suggestions = [x.upper() for x in suggestions]
        else:
            suggestions = [x.lower() for x in suggestions]
        # deal with start of sentence alerts
        if ann.start == 0:
            # deal with insertions
            if ann.end == 0:
                first_word = ann_tokens._tokens[0]
                is_name = first_word in names and first_word.lower() not in words
                if not is_name and not is_all_upper:
                    first_word = first_word.lower()
                suggestions = [" ".join([x, first_word]) for x in suggestions]
                if not is_all_upper:
                    suggestions = [smart_capitalize(x) for x in suggestions]
                new_ann = Annotation(0, 1, ann.source_text, suggestions, ann.meta)
            # deal with deletions
            elif first_suggestion == "":
                end_pos = ann.end + 1
                suggestion = " ".join(ann_tokens._tokens[ann.end:end_pos])
                if not is_all_upper:
                    suggestion = smart_capitalize(suggestion)
                new_ann = Annotation(ann.start, end_pos, ann.source_text,
                                     suggestion, ann.meta)
            # deal with confused
            else:
                if not is_all_upper:
                    suggestions = [smart_capitalize(x) for x in suggestions]
                new_ann = Annotation(ann.start, ann.end, ann.source_text,
                                     suggestions, ann.meta)
        # deal with all other cases
        else:
            new_ann = Annotation(ann.start, ann.end, ann.source_text,
                                 suggestions, ann.meta)
        new_ann_tokens.annotate(new_ann.start, new_ann.end,
                                new_ann.suggestions, new_ann.meta)
    new_ann_sent = AnnotatedText(new_ann_tokens.get_annotated_text())
    return new_ann_sent


def fix_opc_output(ann_sent):
    for ann in ann_sent.iter_annotations():
        if not ann.suggestions:
            ann_sent.remove(ann)
    return ann_sent


def get_system_response(sent, system_type, error_type):
    if system_type == "Patterns":
        ann_sent = check(sent)[0]
        ann_sent = fix_patterns_output(ann_sent)
    elif system_type == "OPC":
        filters = False
        try:
            ann_sent = opc_check(sent, addr='PREPROD', filters=filters)
            ann_sent = fix_opc_output(ann_sent)
        except ApiError:
            ann_sent = AnnotatedText(sent)
    elif system_type == "OPC-filtered":
        try:
            ann_sent = opc_check(sent, addr='PREPROD')
            ann_sent = fix_opc_output(ann_sent)
        except ApiError:
            ann_sent = AnnotatedText(sent)
    elif system_type == "UPC":
        # upc_addr = "upc-high-recall-server.phantasm.gnlp.io:8081"
        # ann_sent = upc_check(sent, addr=upc_addr, custom_server=True)
        ann_sent = upc_check(sent)
        try:
            prev_ann_sent = ann_sent
            ann_sent = fix_upc_output(ann_sent)
        except Exception as e:
            print(e)
            print(f"There is a problem with "
                  f"{prev_ann_sent.get_annotated_text()}.\nNew ann sent"
                  f" looks like {ann_sent.get_annotated_text()}")
    else:
        raise Exception("Unknown system type")
    ann_sent = AnnotatedTokens(ann_sent)
    for ann in ann_sent.iter_annotations():
        ann.meta['system_type'] = system_type
        # do filtering
        if error_type is not None:
            norm_error_type = get_normalized_error_type(ann)
            if error_type != norm_error_type:
                ann_sent.remove(ann)
    return ann_sent


def get_protected_response(sent, system_type, error_type):
    response = None
    while not response:
        try:
            response = get_system_response(sent, system_type, error_type)
        except Exception as e:
            print(e)
            print(sent)
            print(system_type)
            print("Something went wrong. Sleep for 5 sec.")
            sleep(5)
    return response


def get_all_submissions(gold_sent, error_type=None, system_types=None):
    gold_ann = AnnotatedTokens(AnnotatedText(gold_sent))
    orig_sent = gold_ann.get_original_text()
    empty_ann = AnnotatedTokens(AnnotatedText(orig_sent))
    output_records = [gold_ann]
    for st in ["Patterns", "UPC", "OPC"]:
        if system_types is not None and st not in system_types:
            cur_ann = empty_ann
        else:
            cur_ann = get_protected_response(orig_sent, st, error_type)
        output_records.append(cur_ann)
    return output_records


def collect_other_anns_list(paired_anns, selected_idx):
    other_anns_list = []
    for j, list_of_anns in enumerate(paired_anns):
        if j != selected_idx:
            other_anns_list.extend(list_of_anns)
    return other_anns_list


def collect_all_features(gold_sent, error_type=None, system_types=None):
    """Get one sentence which combined outputs of all systems"""
    gold_ann, patterns_ann, upc_ann, opc_ann = get_all_submissions(gold_sent,
                                                                   error_type=None,
                                                                   system_types=system_types)
    all_anns = []
    tokens = gold_ann._tokens
    for ann_sent in [patterns_ann, upc_ann, opc_ann]:
        all_anns.extend(ann_sent.get_annotations())
    paired_anns = split_annotations_on_disjoint_pairs(all_anns)
    result_sentences = []

    ann_dict = {}
    for system_name in ['Patterns', 'OPC', 'UPC']:
        ann_dict[system_name] = AnnotatedTokens(tokens)
    for i, comb_anns_list in enumerate(paired_anns):
        other_anns_list = collect_other_anns_list(paired_anns, i)
        for ann in comb_anns_list:
            new_ann_sent = AnnotatedTokens(tokens)
            new_ann = modify_ann(new_ann_sent, gold_ann, ann,
                                 comb_anns_list, other_anns_list)
            new_ann_type = get_normalized_error_type(new_ann)
            if error_type is not None and new_ann_type != error_type:
                continue
            # add annotation to raw sentence
            new_ann_sent.annotate(new_ann.start, new_ann.end,
                                  new_ann.suggestions, new_ann.meta)
            # add annotation to corresponding system
            ann_dict[ann.meta['system_type']].annotate(new_ann.start,
                                                       new_ann.end,
                                                       new_ann.suggestions,
                                                       meta=new_ann.meta,
                                                       on_overlap=OnOverlap.OVERRIDE)
            result_sentences.append(new_ann_sent.get_annotated_text(
                with_meta=True))
    return result_sentences, ann_dict


def main(args):
    gold_sentences = read_lines(args.gold_file)
    out_file = args.gold_file.replace(".txt", "_with_features.txt")
    et = args.error_type
    if args.system_type is not None:
        st = args.system_type.split()
    else:
        st = None
    for i in tqdm(range(int(len(gold_sentences)/args.chunk_size) + 1)):
        chunk = [[x, et, st] for x in gold_sentences[i*args.chunk_size: (i+1)*args.chunk_size]]
        if not chunk:
            continue
        with ThreadPoolExecutor(args.n_threads) as pool:
            processed_chunk = pool.map(sentence_processor_wrapper, chunk)
        processed_chunk = [[x[0], x[1]] for x in processed_chunk]
        out_dict = {}
        for system_name in ['Patterns', 'OPC', 'UPC']:
            out_dict[system_name] = [x[1][system_name] for x in processed_chunk]
        processed_chunk = [x[0] for x in processed_chunk if x]
        lines_with_anns = []
        for elem in processed_chunk:
            lines_with_anns.extend(elem)
        if lines_with_anns:
            write_lines(out_file, lines_with_anns, mode='a')
            print(f'{len(lines_with_anns)} sents were dumped.')
        if out_dict:
            for system_name, sentences in out_dict.items():
                out_file_tmp = out_file.replace(".txt", f"_{system_name}.txt")
                write_lines(out_file_tmp, sentences, mode='a')


if __name__ == '__main__':
    print('Start feature mapping')
    parser = argparse.ArgumentParser()
    parser.add_argument('gold_file',
                        help='Path to the gold file with annotations')
    parser.add_argument('--error_type',
                        help='Set if you want to filter error only from '
                             'one error type.',
                        default=None)
    parser.add_argument('--system_type',
                        help='Set if you want to collect features only from '
                             'one system.',
                        default=None)
    parser.add_argument('--n_threads',
                        type=int,
                        help='The number of threads.',
                        default=100)
    parser.add_argument('--chunk_size',
                        type=int,
                        help='The size of processing chunks.',
                        default=10000)
    args = parser.parse_args()
    main(args)


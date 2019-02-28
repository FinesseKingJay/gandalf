from scorer.helpers.error_types_bank import ErrorTypesBank
from grampy.text import AnnotatedText, AnnotatedTokens


def filter_by_error_type(raw_list, error_type, default_system_type=None,
                         with_meta=False):
    ebank = ErrorTypesBank()
    output = []
    cnt_target_errors = 0
    cnt_other_errors = 0
    if default_system_type is not None:
        system_type = default_system_type
    for sent in raw_list:
        ann_tokens = AnnotatedTokens(AnnotatedText(sent))
        for ann in ann_tokens.iter_annotations():
            if 'error_type' in ann.meta:
                ann_error_type = ann.meta['error_type']
            elif 'pname' in ann.meta:
                ann_error_type = ann.meta['pname']
            else:
                print(f'Broken annotation {ann}')
                ann_error_type = "OtherError"
            # set system type
            if default_system_type is None:
                system_type = ann.meta['system_type']
            if system_type.startswith('OPC'):
                norm_error_type = ebank.opc_to_patterns22(ann_error_type)
            elif system_type.startswith('UPC'):
                norm_error_type = ebank.upc5_to_patterns22(ann_error_type)
            elif system_type == 'Patterns':
                norm_error_type = \
                    ebank.pname_to_patterns22(ann_error_type)
            elif system_type == "CLC":
                norm_error_type = ebank.clc89_to_patterns22(ann_error_type)
            else:
                print(f'Unknown system {system_type}')
                norm_error_type = "OtherError"
            if norm_error_type != error_type:
                ann_tokens.remove(ann)
                cnt_other_errors += 1
            else:
                cnt_target_errors += 1
        output.append(ann_tokens.get_annotated_text(with_meta=with_meta))
    print(f'Stats: N_target_errors = {cnt_target_errors}, '
          f'N_other_errors = {cnt_other_errors}')
    return output, cnt_target_errors


def annotation_contains_no_suggestions(anno_text):
    for anns in AnnotatedText(anno_text).get_annotations():
        if len(anns.suggestions) == 0:
            return True
    return False


def filter_by_nosuggestions_in_gold(gold_annotations, orig_lines):
    gold_annotations_filtered = list()
    orig_lines_filtered = list()
    for gold_anno_text, orig_sentence in zip(gold_annotations, orig_lines):
        if annotation_contains_no_suggestions(gold_anno_text):
            continue
        gold_annotations_filtered.append(gold_anno_text)
        orig_lines_filtered.append(orig_sentence)
    return gold_annotations_filtered, orig_lines_filtered

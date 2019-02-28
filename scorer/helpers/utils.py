import os


def read_lines(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [s.strip() for s in lines if s]


def write_lines(fn, lines, mode='w'):
    if mode == 'w' and os.path.exists(fn):
        os.remove(fn)
    with open(fn, encoding='utf-8', mode=mode) as f:
        f.writelines(['%s\n' % s for s in lines])


def is_lists_intersection(list1, list2):
    return len(set(list1).intersection(set(list2))) > 0


def startswith_like_list(target_string, list_of_strings):
    for curr_string in list_of_strings:
        if target_string.startswith(curr_string):
            return True
    return False

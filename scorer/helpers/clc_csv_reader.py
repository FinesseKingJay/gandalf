import pandas as pd


class ClcCsvReader():
    """ClcCsvReader reads CLC dataset from CSV using Pandas."""
    def __init__(self, fn):
        self.df = pd.read_csv(fn, low_memory=False)

    def iter_items(self, max_item_number=None):
        if max_item_number is None:
            max_item_number = float('Inf') # typing 0 in command line is much more easier than "Inf"
        for k, row in enumerate(self.df.itertuples(index=True)):
            if k == 0: continue
            if k > max_item_number: break
            sample_id = row[0]
            orig = getattr(row, 'orig')
            corr = getattr(row, 'corr')
            anno_text = getattr(row, 'anno_text')
            relabeled_anno_text = getattr(row, 'relabeled_anno_text')
            err_types = str(getattr(row, 'err_types'))
            error_types_list = list()
            if err_types != 'nan' and len(err_types) > 0:
                error_types_list = err_types.split(',')
            yield sample_id, orig, corr, anno_text, relabeled_anno_text, error_types_list

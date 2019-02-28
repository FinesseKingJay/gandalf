class Batcher():
    def __init__(self, list_in, batch_size, verbose=False):
        self.list_in = list_in
        self.batch_size = batch_size
        self.verbose = verbose
        self.data_num = len(list_in)

    def iter_batches(self):
        batch_num = self.data_num // self.batch_size + 1
        for n in range(batch_num):
            if self.verbose:
                print('\n--- processing batch %d / %d ---\n' % (n + 1, batch_num))
            i = n*self.batch_size
            j = min((n+1)*self.batch_size, self.data_num)
            batch = [self.list_in[k] for k in range(i, j)]
            yield batch

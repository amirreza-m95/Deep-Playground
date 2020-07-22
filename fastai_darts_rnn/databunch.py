# src : https://github.com/tinhb92/rnn_darts_fastai/blob/master/databunch_nb.ipynb
from fastai import *
from fastai.text import *

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = collections.Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return token_id

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        self.num_tokens = len(self.dictionary)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
    
# https://github.com/fastai/fastai_docs/blob/master/dev_nb/007_wikitext_2.ipynb
# download data from https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
EOS = '<eos>'
PATH=Path('data/penn')

def read_file(filename):
    tokens = []
    with open(PATH/filename, encoding='utf8') as f:
        for line in f:
            tokens.append(line.split() + [EOS])
    return np.array(tokens)

train_tok = read_file('train.txt')
valid_tok = read_file('valid.txt')
test_tok = read_file('test.txt')

cor = Corpus('data/penn')
voc = Vocab(cor.dictionary.idx2word)

bs = 256
bptt = 35

dat = TextLMDataBunch.from_tokens('data', 
                                  trn_tok = train_tok, trn_lbls = None,
                                  val_tok = valid_tok, val_lbls=None,
                                  tst_tok = test_tok, 
                                  vocab = voc,
                                  bs = bs, bptt = bptt)

dat.save('penn_db')
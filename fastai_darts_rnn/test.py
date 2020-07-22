
# src: https://github.com/tinhb92/rnn_darts_fastai/blob/master/test_nb.ipynb


# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

import fastai
print(fastai.__version__)
from fastai import *
from fastai.text import *
from fastai.callbacks import CSVLogger, SaveModelCallback

import torch.backends.cudnn as cudnn
from collections import namedtuple

from train import DartsRnn, ASGD_Switch
from darts_callbacks import HidInit, Regu, SaveModel, ResumeModel, Genotype

# random seed for reproducibility.
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

gpu = 0
torch.cuda.set_device(gpu)
cudnn.benchmark = True
cudnn.enabled=True
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(seed)

bs_train, bs_val, bs_test = 64, 10, 1
bptt = 35
dat = load_data('data', 'penn_db', bs=bs_train, bptt = bptt)
dat.valid_dl.batch_size = bs_val
dat.test_dl.batch_size = bs_test
vocab_sz = len(dat.train_ds.x.vocab.itos)
emb_sz = 850
hid_sz = 850
wdecay = 8e-7
dropout = 0.75
dropouth = 0.25
dropoutx = 0.75
dropouti = 0.2
dropoute = 0.1
clip = 0.25
nonmono = 5


# input genotype for DartsRnn
DARTS_V1 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('tanh', 2), ('relu', 3), ('relu', 4), 
                               ('identity', 1), ('relu', 5), ('relu', 1)], concat=range(1, 9))

DARTS_V2 = Genotype(recurrent=[('sigmoid', 0), ('relu', 1), ('relu', 1), ('identity', 1), ('tanh', 2), 
                               ('sigmoid', 5), ('tanh', 3), ('relu', 5)], concat=range(1, 9))


learn = Learner(dat, DartsRnn(emb_sz = emb_sz, vocab_sz = vocab_sz,
                              ninp = emb_sz, nhid = hid_sz, 
                              dropout = dropout, dropouth = dropouth, dropoutx = dropoutx,
                              dropouti = dropouti, dropoute = dropoute,
                              bs_train = bs_train, bs_val = bs_val, bs_test = bs_test,
                              genotype = DARTS_V1),
                opt_func = torch.optim.SGD,
                callback_fns = [
                    HidInit,
                ],
                wd = wdecay, true_wd=False
                )

total_params = sum(x.nelement() for x in learn.model.parameters())
print(total_params)
print(learn.model.rnn.genotype)




def resu(learn, name):
    checkpoint = torch.load(learn.path/learn.model_dir/f'{name}.pth', 
                            map_location=lambda storage, loc: storage)
    learn.model.load_state_dict(checkpoint['model'])
    return learn

resume_model = 'darts_V1'
learn = resu(learn, resume_model)



learn.model.test = True
learn.validate(learn.data.test_dl)


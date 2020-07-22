# https://github.com/tinhb92/rnn_darts_fastai/blob/master/train_nb.ipynb

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

import fastai
print(fastai.__version__)
from fastai import *
from fastai.text import *
from fastai.callbacks import CSVLogger, SaveModelCallback

import torch.backends.cudnn as cudnn

from train import DartsRnn, ASGD_Switch
from darts_callbacks import HidInit, Regu, SaveModel, ResumeModel, GcCol, Genotype


# random seed for reproducibility.
seed = 8
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

gpu = 0
torch.cuda.set_device(gpu)
cudnn.benchmark = True
cudnn.enabled=True
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(seed)

bs_train, bs_val = 64, 10
bptt = 35
dat = load_data('data', 'penn_db', bs=bs_train, bptt=bptt)
dat.valid_dl.batch_size = bs_val
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


# genotype for DartsRnn
DARTS_V1 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('tanh', 2), ('relu', 3), ('relu', 4), 
                               ('identity', 1), ('relu', 5), ('relu', 1)], concat=range(1, 9))

DARTS_V2 = Genotype(recurrent=[('sigmoid', 0), ('relu', 1), ('relu', 1), ('identity', 1), ('tanh', 2), 
                               ('sigmoid', 5), ('tanh', 3), ('relu', 5)], concat=range(1, 9))


csv_name = 'train'
model_name = 'train'
resume_model = ''


learn = Learner(dat, DartsRnn(emb_sz = emb_sz, vocab_sz = vocab_sz,
                              ninp = emb_sz, nhid = hid_sz, 
                              dropout = dropout, dropouth = dropouth, dropoutx = dropoutx,
                              dropouti = dropouti, dropoute = dropoute,
                              bs_train = bs_train, bs_val = bs_val,
                              genotype = DARTS_V1), 
                opt_func = torch.optim.SGD,
                callback_fns = [
                    HidInit,
                    Regu,
                    partial(GradientClipping, clip=clip),
                    partial(CSVLogger, filename = csv_name, append=True),
                    partial(ASGD_Switch, nonmono = nonmono, asgd=False), 
                    # asgd == True if using asgd right from the start
                    GcCol,
#                     partial(ResumeModel, name = resume_model)
                ],
                wd = wdecay, true_wd=False
                )

total_params = sum(x.nelement() for x in learn.model.parameters())
print(total_params)
print(learn.model.rnn.genotype)



learn.fit(5, 20, callbacks=[
#     SaveModel(learn, gap = 20, name=model_name),
    SaveModelCallback(learn, name=model_name) # save on improvement
                              ])



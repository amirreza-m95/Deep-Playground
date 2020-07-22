
# https://github.com/tinhb92/rnn_darts_fastai/blob/master/train_search_nb.ipynb

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai import *
from fastai.text import *
from fastai.callbacks import CSVLogger, SaveModelCallback

import torch.backends.cudnn as cudnn

from train_search import DartsRnnSearch, ArchParamUpdate, PrintGenotype
from darts_callbacks import HidInit, Regu, SaveModel, ResumeModel, GcCol

# random seed for reproducibility.
seed = 135
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

gpu = 0
torch.cuda.set_device(gpu)
cudnn.benchmark = True
cudnn.enabled=True
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(seed)

bptt = 35
bs_train, bs_val = 256, 64
dat = load_data('data', 'penn_db', bs=bs_train, bptt=bptt)
dat.valid_dl.batch_size = bs_val
search_dat = list(iter(load_data('data', 'penn_db', bs=bs_train, bptt = bptt).valid_dl))

arch_lr = 3e-3
arch_wdecay = 1e-3
vocab_sz = len(dat.train_ds.x.vocab.itos)
emb_sz = 300
hid_sz = 300
wdecay = 5e-7
dropout = 0.75
dropouth = 0.25
dropoutx = 0.75
dropouti = 0.2
dropoute = 0.
clip = 0.25

csv_name = 'train_search'
model_name = 'train_search'
# resume_model = 'train_search' 

learn = Learner(dat, DartsRnnSearch(emb_sz = emb_sz, vocab_sz = vocab_sz,
                                    ninp = emb_sz, nhid = hid_sz, 
                                    dropout = dropout, dropouth = dropouth, dropoutx = dropoutx,
                                    dropouti = dropouti, dropoute = dropoute,
                                    bs_train = bs_train, bs_val = bs_val),
                opt_func = torch.optim.SGD,
                callback_fns = [
                    HidInit,
                    partial(ArchParamUpdate, search_dat=search_dat,
                            arch_lr=arch_lr, arch_wdecay=arch_wdecay, wdecay=wdecay),
                    Regu,
                    PrintGenotype,
                    partial(GradientClipping, clip=clip),
                    partial(CSVLogger, filename = csv_name, append = True),
                    GcCol,
#                     partial(ResumeModel, name = resume_model) 
                ], 
                wd = wdecay
                )

total_params = sum(x.nelement() for x in learn.model.parameters())
print('Total params:', total_params)
print(learn.model.genotype_parse())
# learn.data.valid_dl=None


learn.fit(50, 20, callbacks=[
    SaveModel(learn, gap = 1, name=model_name),
    SaveModelCallback(learn, name=model_name) # save on improvement
                           ])
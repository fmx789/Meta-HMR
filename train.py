import os
import random
import numpy as np
import torch
from utils import TrainOptions
from train import Trainer_cliff, Trainer_hmr

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    if options.model == 'cliff':
        trainer = Trainer_cliff(options)
    elif options.model == 'hmr':
        trainer = Trainer_hmr(options)
    trainer.train()

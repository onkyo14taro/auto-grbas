from argparse import ArgumentParser
import os
from pathlib import Path
import sys

import torch
from pytorch_lightning import Trainer


PARENT_DIR = os.path.dirname(os.path.abspath(__file__))


os.chdir(PARENT_DIR)
sys.path.append(PARENT_DIR)


from models import BaseModel, run


def main():
    ######################################################################
    ### Hyperparameter setting
    ######################################################################
    parser = ArgumentParser()


    ### add PROGRAM level args
    ######################################################################

    ### add experiment level args
    ######################################################################
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--results_dir',
                        default=str(Path(PARENT_DIR).parent/"results"/"raw"))

    ### add model specific args
    ######################################################################
    parser = BaseModel.add_model_specific_args(parser)


    ### add all the available trainer options to argparse
    ### ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    ######################################################################
    parser = Trainer.add_argparse_args(parser)


    ### parse
    ######################################################################
    hparams = parser.parse_args()
    if not torch.cuda.is_available():
        hparams.gpus = 0
        hparams.precision = 32

    ### force settings for reproducibility
    ######################################################################
    hparams.deterministic = True
    hparams.benchmark = False

    # To make RNN deterministic
    # See https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


    ######################################################################
    ### run the experiment
    ######################################################################
    run(hparams)


if __name__ == '__main__':
    main()

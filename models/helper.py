import copy
import time

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class _RunManager:
    def __init__(self, epochs: int, hparams_dict: dict, n_statevars: int, n_rules: int, n_input: int, patience: int = 10, delta: int = 0.0001):
        """Run Manager keeps track of epochs, (best) losses and prints the progress bars. Also controls the tensorboard.

        Args:
            epochs (int): No. of epochs.
            hparams_dict (dict): (Additional) hyperparameters to store in tensorboard.
            n_statevars (int): No. of state variables.
            n_rules (int): No. of rules.
            n_input (int): No. of inputs.
            patience (int, optional): Patience parameter. Defaults to 10.
            delta (int, optional): Delta parameter. Defaults to 0.0001.
        """

        # sanfis parameters
        self.best_weights = None
        self.epochs = epochs
        self.hparams_dict = hparams_dict
        self.n_statevars = n_statevars
        self.n_input = n_input
        self.n_rules = n_rules

        # early stopping criteria
        self.patience = patience
        self.iter = 0
        self.counter = 0
        self.best_loss = float('inf')
        self.global_best_loss = float('inf')
        self.early_stop = False
        self.delta = delta

        # train and valid curve
        self.train_curve_iter = []     # train loss per iteration
        self.train_curve = []    # train loss per epoch
        self.valid_curve = []    # valid loss per epoch

        # epoch counter
        self.epoch = 0

        # progress bar
        self.start_time = time.time()
        self.pbar_step = self.epochs / 100
        self.tbwriter = None

    def __call__(self, model_weights, epoch, train_loss, valid_loss, pbar):
        self.epoch += 1

        # track losses
        self._track_losses(train_loss, valid_loss)

        loss = self.valid_curve[-1].item()

        # check early stop criteria
        if loss + self.delta < self.best_loss:
            self.best_loss = loss
            self.epoch = epoch
            self.counter = 0
            if loss < self.global_best_loss:
                self.global_best_loss = loss
                self.save_checkpoint(model_weights)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        # update progress bar
        if self.epoch % self.pbar_step == 0:
            pbar.update(self.pbar_step)
            pbar.set_postfix(
                train_loss=round(self.train_curve[-1].item(), 5), valid_loss=round(self.valid_curve[-1].item(), 5))

    def get_writer(self, logdir):
        if logdir == None:
            logdir = 'logs/runs/'

        logDATE = __import__("datetime").datetime.now().strftime(
            '%Y_%m_%d_%H%M%S')

        logHPARAM = ''.join(
            [f'_{d}{self.hparams_dict[d]}' for d in self.hparams_dict])
        logNAME = f'-S{self.n_statevars}_N{self.n_input}_R{self.n_rules}{logHPARAM}'

        writer = SummaryWriter(logdir + logDATE + logNAME, comment=logNAME)
        # writer = SummaryWriter(comment=logNAME) # alternative

        self.tbwriter = writer

    def save_checkpoint(self, weights):
        self.best_weights = copy.deepcopy(weights)

    def load_checkpoint(self):
        return self.best_weights

    def reset_earlystopper(self):
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def _track_losses(self, train_loss, valid_loss):
        self.train_curve.append(sum(train_loss) / len(train_loss))
        self.train_curve_iter.extend(train_loss)
        self.valid_curve.append(sum(valid_loss) / len(valid_loss))

        if self.tbwriter:
            self.tbwriter.add_scalar(
                'Loss/train', self.train_curve[-1], self.epoch)
            self.tbwriter.add_scalar(
                'Loss/valid', self.valid_curve[-1], self.epoch)
            # #  add histograms of weights
            # for name, weight in self.named_parameters():
            #     writer.add_histogram(name, weight, epoch)

    def end_training(self):
        self.run_time = time.time() - self.start_time

        if self.tbwriter:
            # log hparams
            HPARAMS = {
                       "n_statevars": self.n_statevars,
                       "n_input": self.n_input,
                       "n_rules": self.n_rules,
                       **self.hparams_dict}
            self.tbwriter.add_hparams(HPARAMS,
                                      {
                                          "train_loss": self.train_curve[-1],
                                          "valid_loss": self.global_best_loss,
                                      },
                                      )
            # close writer
            self.tbwriter.flush()
            self.tbwriter.close()

    def get_report_history(self):

        report = {**self.hparams_dict,
                  "n_statevars": self.n_statevars,
                  "n_input": self.n_input,
                  "n_rules": self.n_rules,
                  "run_time": self.run_time}

        history = pd.DataFrame({'train_curve': np.array(
            self.train_curve), 'valid_curve': np.array(self.valid_curve)}).rename_axis('epoch')

        return report, history
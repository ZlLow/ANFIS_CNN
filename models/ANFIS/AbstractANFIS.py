import numpy as np
import pandas as pd
import torch

from abc import ABC, abstractmethod
from typing import Optional, Union, Callable, List

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import TensorDataset
from tqdm import tqdm

from models.helper import _RunManager


class AbstractANFIS(ABC, nn.Module):
    def __init__(self, n_input: int, membfuncs: Optional[list], to_device: Optional[str] = None,
                 drop_out_rate: Optional[int] = 0.2):
        super().__init__()
        self.report = None
        self.optimizer = None
        self._initial_premise = None
        self._membfuncs = membfuncs
        self._s = len(membfuncs) if membfuncs else 0
        self._memberships = [memb['n_memb'] for memb in membfuncs] if membfuncs else []
        self._rules: int = int(np.prod(self._memberships)) if self._memberships else 0
        self._n = n_input
        self.scaler = MinMaxScaler()
        self.drop_out_rate = drop_out_rate

        if to_device is None:
            self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = to_device

        self.to(self.device)

    @abstractmethod
    def _reset_model_parameter(self):
        pass

    def fit(self, train_data: TensorDataset, valid_data: TensorDataset, optimizer: Optional[Optimizer],
            loss_function: Callable, batch_size: int = 27, epochs: int = 100,
            hparams_dict: dict = {}) -> pd.DataFrame:

        self.optimizer = optimizer

        train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
        valid_dl = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)

        # setup run manager
        run_manager = _RunManager(epochs, hparams_dict, self.n_statevars, self.num_rules, self._n)

        with tqdm(total=epochs, desc="Training Loop", unit='epochs') as pbar:
            for epoch in range(epochs):
                train_loss = []
                self.train()

                for batch_data in train_dl:
                    # Handle different data formats (2 or 3 tensors)
                    if len(batch_data) == 2:
                        xb_train, yb_train = batch_data
                        sb_train = xb_train  # Use same data for both inputs
                    else:
                        sb_train, xb_train, yb_train = batch_data

                    sb_train = sb_train.to(self.device)
                    xb_train = xb_train.to(self.device)
                    yb_train = yb_train.to(self.device)

                    train_pred = self(sb_train, xb_train)
                    loss = loss_function(train_pred, yb_train)
                    train_loss.append(loss)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                with torch.no_grad():
                    self.eval()
                    valid_loss = []

                    # Handle validation data format
                    for valid_batch_data in valid_dl:
                        if len(valid_batch_data) == 2:
                            xb_valid, yb_valid = valid_batch_data
                            sb_valid = xb_valid.to(self.device)
                        else:
                            sb_valid, xb_valid, yb_valid = valid_batch_data

                        y_pred_valid = self(sb_valid.to(self.device), xb_valid.to(self.device))
                        valid_loss.append(loss_function(y_pred_valid, yb_valid.to(self.device)))
                        run_manager(self.state_dict(), epoch, train_loss, valid_loss, pbar)

                        if run_manager.early_stop:
                            self._reset_model_parameter()
                            run_manager.reset_earlystopper()

        best_weight = run_manager.load_checkpoint()
        self.load_state_dict(best_weight)


        run_manager.end_training()
        self.report, history = run_manager.get_report_history()
        return history

    def predict(self, input_data: Union[TensorDataset, List[torch.Tensor]], batch_size: int = 1000) -> torch.Tensor:
        if isinstance(input_data, TensorDataset):
            dataloader = torch.utils.data.DataLoader(input_data, batch_size=batch_size, shuffle=False)
        else:
            # Handle list of tensors
            if len(input_data) == 1:
                dataset = TensorDataset(input_data[0], input_data[0])
            else:
                dataset = TensorDataset(*input_data)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            self.eval()
            y_pred_scaled = torch.tensor([])

            for batch_data in dataloader:
                if len(batch_data) == 2:
                    sb, xb = batch_data
                else:
                    sb, xb, _ = batch_data

                pred = self(sb.to(self.device), xb.to(self.device))
                y_pred_scaled = torch.cat((y_pred_scaled, pred.cpu())).reshape(-1, 1)

        y_pred = y_pred_scaled
        return y_pred

    def plotmfs(self, show_initial_weights=True, show_firingstrength: bool = True, bounds: Optional[list] = None,
                names: Optional[list] = None, title: Optional[str] = None, show_title: bool = True,
                save_path: Optional[str] = None):
        """Plots the membership functions."""
        if not hasattr(self, 'layers') or 'fuzzylayer' not in self.layers:
            print("No fuzzy layer found in this model. Plotting not available for clustered input models.")
            return

        # plot bounds
        if not bounds:
            lower_s = self.scaler.data_min_
            higher_s = self.scaler.data_max_
        else:
            lower_s, higher_s = list(zip(*bounds))

        # (scaled) state variables
        SN = torch.empty((1000, self._s))
        for i, (smin, smax) in enumerate(zip(lower_s, higher_s)):
            SN[:, i] = torch.linspace(smin, smax, 1000)
        SN_scaled = self.scaler.transform(SN)

        # membership curves
        with torch.no_grad():
            membership_curves = []
            for i, layer in enumerate(self.layers.fuzzylayer.fuzzyfication):
                membership_curves.append(
                    layer(SN_scaled[:, [i]]).detach().numpy())

        # initial membership curves
        with torch.no_grad():
            init_membership_curves = []
            for i, layer in enumerate(_MFLayer(self._initial_premise).fuzzyfication):
                init_membership_curves.append(
                    layer(SN_scaled[:, [i]]).detach().numpy())

        # set plot names
        if names == None:
            plot_names = [
                f'State Variable {s + 1} ({self.premise[s]["function"]})' for s in range(self.n_statevars)]
        else:
            plot_names = names

        # setup plot
        fig, ax = plt.subplots(
            nrows=self.n_statevars, ncols=1, figsize=(8, self.n_statevars * 3))
        if show_title:
            if title == None:
                fig.suptitle(f'Membership functions', size=16)
            else:
                fig.suptitle(title, size=16)
        fig.subplots_adjust(hspace=0.4)

        # plot curves
        for s, curve in enumerate(membership_curves):
            ax[s].grid(True)
            ax[s].set_title(
                plot_names[s], size=19)
            # prepare colors
            colors = []
            for m in range(curve.shape[1]):
                colors.append(next(ax[s]._get_lines.prop_cycler)['color']
                              )
            # plot membfuncs for each statevar
            for m in range(curve.shape[1]):
                ax[s].plot(SN[:, s], curve[:, m], color=colors[m])
                if show_initial_weights:
                    ax[s].plot(SN[:, s], init_membership_curves[s][:, m],
                               '--', color=colors[m], alpha=.5)
            # show normalized memb_funcs
            if show_firingstrength:
                norm_curve = curve / curve.sum(1).reshape(-1, 1)
                ax[s].stackplot(
                    SN[:, s], [col for col in norm_curve.T], alpha=0.3, colors=colors)

            ax[s].tick_params(axis='x', labelsize=14)
            ax[s].tick_params(axis='y', labelsize=14)

        plt.show()

        if save_path != None:
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    @property
    def n_statevars(self) -> int:
        return self._s

    @property
    def n_input(self):
        return self._n

    @property
    def memberships(self):
        return self._memberships

    @property
    def num_rules(self):
        return self._rules

    @property
    def scaling_params(self):
        return self.scaler.__dict__


class _MFLayer(nn.Module):
    def __init__(self, membfuncs):
        super(_MFLayer, self).__init__()
        self.n_statevars = len(membfuncs)

        fuzzyification = nn.ModuleList()

        for mf in membfuncs:
            if mf['function'] == 'gaussian':
                MembershipLayer = _GaussianFuzzyLayer(mf['params'], mf['n_memb'])
            elif mf['function'] == 'bell':
                MembershipLayer = _BellFuzzyLayer(mf['params'], mf['n_memb'])
            elif mf['function'] == 'sigmoid':
                MembershipLayer = _SigmoidFuzzyLayer(mf['params'], mf['n_memb'])
            else:
                raise NotImplementedError

            fuzzyification.append(MembershipLayer)

        self.fuzzyification = fuzzyification

    def reset_parameters(self):
        [layer.reset_parameters() for layer in self.fuzzyification]

    def forward(self, x):
        output = [Layer(x[:, [i]]) for i, Layer in enumerate(self.fuzzyification)]
        return output


class _GaussianFuzzyLayer(nn.Module):
    def __init__(self, params: dict, n_memb: int):
        """Represents the gaussian fuzzy layer (layer 1) of anfis. Inputs will be fuzzyfied
        """
        super(_GaussianFuzzyLayer, self).__init__()
        self.params = params
        self.m = n_memb

        self._mu = torch.tensor([params['mu']['value']])
        self._sigma = torch.tensor([params['sigma']['value']])

        if params['mu']['trainable']:
            self._mu = nn.Parameter(self._mu)

        if params['sigma']['trainable']:
            self._sigma = nn.Parameter(self._sigma)

    @property
    def coeffs(self):
        return {'function': 'gaussian',
                'n_memb': self.m,
                'params': {'mu': {'value': self._mu.data.clone().flatten().tolist(),
                                  'trainable': isinstance(self._mu, nn.Parameter)},
                           'sigma': {'value': self._sigma.data.clone().flatten().tolist(),
                                     'trainable': isinstance(self._sigma, nn.Parameter)}
                           }
                }

    def reset_parameters(self):
        with torch.no_grad():
            self._mu[:] = torch.tensor([self.params['mu']['value']])
            self._sigma[:] = torch.tensor([self.params['sigma']['value']])

    def forward(self, input_):
        output = torch.exp(
            - torch.square(
                (input_.repeat(
                    1, self.m).reshape(-1, self.m) - self._mu)
                / self._sigma.square()
            )
        )
        return output


class _BellFuzzyLayer(nn.Module):
    def __init__(self, params: dict, n_memb: int):
        """Represents the bell-shaped fuzzy layer (layer 1) of S-ANFIS. Inputs will be fuzzyfied
        """
        super(_BellFuzzyLayer, self).__init__()
        self.params = params
        self.m = n_memb

        self._c = torch.tensor([params['c']['value']])
        self._a = torch.tensor([params['a']['value']])
        self._b = torch.tensor([params['b']['value']])

        if params['a']['trainable']:
            self._a = nn.Parameter(self._a)

        if params['b']['trainable']:
            self._b = nn.Parameter(self._b)

        if params['c']['trainable']:
            self._c = nn.Parameter(self._c)

    @property
    def coeffs(self):
        return {'function': 'bell',
                'n_memb': self.m,
                'params': {'c': {'value': self._c.data.clone().flatten().tolist(),
                                 'trainable': isinstance(self._c, nn.Parameter)},
                           'a': {'value': self._a.data.clone().flatten().tolist(),
                                 'trainable': isinstance(self._a, nn.Parameter)},

                           'b': {'value': self._b.data.clone().flatten().tolist(),
                                 'trainable': isinstance(self._b, nn.Parameter)}
                           }
                }

    def reset_parameters(self):
        with torch.no_grad():
            self._c[:] = torch.tensor([self.params['c']['value']])
            self._a[:] = torch.tensor([self.params['a']['value']])
            self._b[:] = torch.tensor([self.params['b']['value']])

    def forward(self, input_):

        output = 1 / (1 + torch.pow(((input_.repeat(1,
                                                    self.m).view(-1, self.m) - self._c).square() / self._a), self._b))

        return output


class _SigmoidFuzzyLayer(nn.Module):
    """Represents the sigmoid fuzzy layer (layer 1) of s-anfis. Inputs will be fuzzyfied
    """

    def __init__(self, params: dict, n_memb: int):
        super(_SigmoidFuzzyLayer, self).__init__()
        self.params = params
        self.m = n_memb

        self._c = torch.tensor([params['c']['value']])
        self._gamma = torch.tensor([params['gamma']['value']])

        if params['c']['trainable']:
            self._c = nn.Parameter(self._c)
        if params['gamma']['trainable']:
            self._gamma = nn.Parameter(self._gamma)

    @property
    def coeffs(self):
        return {'function': 'sigmoid',
                'n_memb': self.m,
                'params': {'c': {'value': self._c.data.clone().flatten().tolist(),
                                 'trainable': isinstance(self._c, nn.Parameter)},
                           'gamma': {'value': self._gamma.data.clone().flatten().tolist(),
                                     'trainable': isinstance(self._gamma, nn.Parameter)}
                           }
                }

    def reset_parameters(self):
        with torch.no_grad():
            self._c[:] = torch.tensor([self.params['c']['value']])
            self._gamma[:] = torch.tensor([self.params['gamma']['value']])

    def forward(self, input_):

        # = 1 / (1 + e^(- input_))
        output = torch.sigmoid(
            self._gamma * (input_.repeat(1, self.m).view(-1, self.m) - self._c))

        return output

class _RuleLayer(nn.Module):
    def __init__(self):
        """Rule layer / layer 2 of the S-ANFIS network
        """
        super(_RuleLayer, self).__init__()

    def forward(self, input_):
        batch_size = input_[0].shape[0]
        n_in = len(input_)

        if n_in == 2:
            output = input_[0].view(batch_size, -1, 1) * \
                     input_[1].view(batch_size, 1, -1)

        elif n_in == 3:
            output = input_[0].view(batch_size, -1, 1, 1) * \
                     input_[1].view(batch_size, 1, -1, 1) * \
                     input_[2].view(batch_size, 1, 1, -1)

        elif n_in == 4:
            output = input_[0].view(batch_size, -1, 1, 1, 1) * \
                     input_[1].view(batch_size, 1, -1, 1, 1) * \
                     input_[2].view(batch_size, 1, 1, -1, 1) * \
                     input_[3].view(batch_size, 1, 1, 1, -1)

        elif n_in == 5:
            output = input_[0].view(batch_size, -1, 1, 1, 1, 1) * \
                     input_[1].view(batch_size, 1, -1, 1, 1, 1) * \
                     input_[2].view(batch_size, 1, 1, -1, 1, 1) * \
                     input_[3].view(batch_size, 1, 1, 1, -1, 1) * \
                     input_[4].view(batch_size, 1, 1, 1, 1, -1)

        elif n_in == 6:
            output = input_[0].view(batch_size, -1, 1, 1, 1, 1, 1) * \
                     input_[1].view(batch_size, 1, -1, 1, 1, 1, 1) * \
                     input_[2].view(batch_size, 1, 1, -1, 1, 1, 1) * \
                     input_[3].view(batch_size, 1, 1, 1, -1, 1, 1) * \
                     input_[4].view(batch_size, 1, 1, 1, 1, -1, 1) * \
                     input_[5].view(batch_size, 1, 1, 1, 1, 1, -1)
        elif n_in == 7:
            output = input_[0].view(batch_size, -1, 1, 1, 1, 1, 1, 1) * \
                     input_[1].view(batch_size, 1, -1, 1, 1, 1, 1, 1) * \
                     input_[2].view(batch_size, 1, 1, -1, 1, 1, 1, 1) * \
                     input_[3].view(batch_size, 1, 1, 1, -1, 1, 1, 1) * \
                     input_[4].view(batch_size, 1, 1, 1, 1, -1, 1, 1) * \
                     input_[5].view(batch_size, 1, 1, 1, 1, 1, -1, 1) * \
                     input_[6].view(batch_size, 1, 1, 1, 1, 1, 1, -1)
        elif n_in == 8:
            output = input_[0].view(batch_size, -1, 1, 1, 1, 1, 1, 1, 1) * \
                     input_[1].view(batch_size, 1, -1, 1, 1, 1, 1, 1, 1) * \
                     input_[2].view(batch_size, 1, 1, -1, 1, 1, 1, 1, 1) * \
                     input_[3].view(batch_size, 1, 1, 1, -1, 1, 1, 1, 1) * \
                     input_[4].view(batch_size, 1, 1, 1, 1, -1, 1, 1, 1) * \
                     input_[5].view(batch_size, 1, 1, 1, 1, 1, -1, 1, 1) * \
                     input_[6].view(batch_size, 1, 1, 1, 1, 1, 1, -1, 1) * \
                     input_[7].view(batch_size, 1, 1, 1, 1, 1, 1, 1, -1)
        elif n_in == 9:
            output = input_[0].view(batch_size, -1, 1, 1, 1, 1, 1, 1, 1, 1) * \
                     input_[1].view(batch_size, 1, -1, 1, 1, 1, 1, 1, 1, 1) * \
                     input_[2].view(batch_size, 1, 1, -1, 1, 1, 1, 1, 1, 1) * \
                     input_[3].view(batch_size, 1, 1, 1, -1, 1, 1, 1, 1, 1) * \
                     input_[4].view(batch_size, 1, 1, 1, 1, -1, 1, 1, 1, 1) * \
                     input_[5].view(batch_size, 1, 1, 1, 1, 1, -1, 1, 1, 1) * \
                     input_[6].view(batch_size, 1, 1, 1, 1, 1, 1, -1, 1, 1) * \
                     input_[7].view(batch_size, 1, 1, 1, 1, 1, 1, 1, -1, 1) * \
                     input_[8].view(batch_size, 1, 1, 1, 1, 1, 1, 1, 1, -1)
        elif n_in == 10:
            output = input_[0].view(batch_size, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1) * \
                     input_[1].view(batch_size, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1) * \
                     input_[2].view(batch_size, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1) * \
                     input_[3].view(batch_size, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1) * \
                     input_[4].view(batch_size, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1) * \
                     input_[5].view(batch_size, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1) * \
                     input_[6].view(batch_size, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1) * \
                     input_[7].view(batch_size, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1) * \
                     input_[8].view(batch_size, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1) * \
                     input_[9].view(batch_size, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1)
        else:
            raise Exception(
                f"Model Supports only 2,3,4,5,6,7,8,9 or 10 input variables but {n_in} were given.")

        output = output.reshape(batch_size, -1)

        return output


class _ConsequenceLayer(nn.Module):
    def __init__(self, n_input: int, n_rules: int):
        """Consequence layer / layer 4 of the S-ANFIS network
        """
        super(_ConsequenceLayer, self).__init__()
        self.n = n_input
        self.rules = n_rules

        # weights
        self._weight = nn.Parameter(torch.Tensor(self.n, self.rules))
        self._bias = nn.Parameter(torch.Tensor(1, n_rules))
        self.reset_parameters()

    @property
    def coeffs(self):
        return {'bias': self._bias,
                'weight': self._weight}

    @coeffs.setter
    def coeffs(self, new_coeffs: dict):
        assert type(
            new_coeffs) is dict, f'new coeffs should be dict filled with torch parameters, but {type(new_coeffs)} was given.'
        assert self._bias.shape == new_coeffs['bias'].shape and self._weight.shape == new_coeffs['weight'].shape, \
            f"New coeff 'bias' should be of shape {self._bias.shape}, but is instead {new_coeffs['bias'].shape} \n" \
            f"New coeff 'weight' should be of shape {self._weight.shape}, but is instead {new_coeffs['weight'].shape}"

        # transform to torch Parameter if any coeff is of type numpy array:
        if any(type(coeff) == np.ndarray for coeff in new_coeffs.values()):
            new_coeffs = {key: torch.nn.Parameter(torch.from_numpy(
                new_coeffs[key]).float()) for key in new_coeffs}

        # transform to torch Parameter if any coeff is of type torch.Tensor:
        if any(type(coeff) == torch.Tensor for coeff in new_coeffs.values()):
            new_coeffs = {key: torch.nn.Parameter(
                new_coeffs[key].float()) for key in new_coeffs}

        self._bias = new_coeffs['bias']
        self._weight = new_coeffs['weight']

    def reset_parameters(self):
        with torch.no_grad():
            self._weight[:] = torch.rand(
                self.n, self.rules) - 0.5

            self._bias[:] = torch.rand(1, self.rules) - 0.5

    def forward(self, input_, wnorm):
        output = wnorm * (torch.matmul(input_, self._weight) + self._bias)
        return output
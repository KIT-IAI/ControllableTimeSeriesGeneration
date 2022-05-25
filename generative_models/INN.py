import pickle
import time
from typing import Dict

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import torch
import xarray as xr
from pywatts.core.base import BaseEstimator
from pywatts.core.filemanager import FileManager
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class CondNet(nn.Module):
    def __init__(self, cond_features, horizon):
        super().__init__()

        self.condition = nn.Sequential(nn.Linear(cond_features, 2),
                                       nn.Tanh(),
                                       nn.Linear(2, 1),
                                       # nn.ReLU()
                                       )

    def forward(self, c):
        return self.condition(c)


def subnet(ch_in, ch_out):
    return nn.Sequential(nn.Linear(ch_in, 32),
                         nn.Tanh(),
                         # nn.Linear(32,16),
                         # nn.Tanh(),
                         # nn.Linear(16, 8),
                         # nn.Tanh(),
                         # nn.Linear(8,16),
                         # nn.Tanh(),
                         nn.Linear(32, ch_out))


class INN(nn.Module):
    def __init__(self, lr, cond_features, horizon, n_layers_cond=5, n_layers_without_cond=0, subnet=subnet):
        super().__init__()
        self.horizon = horizon
        if cond_features > 0:
            self.cond_net = CondNet(cond_features, horizon)
        else:
            self.cond_net = None

        self.no_layer_cond = n_layers_cond
        self.no_layer_without_cond = n_layers_without_cond
        self.subnet = subnet
        self.cinn = self.build_inn()

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)
        if self.cond_net:
            self.trainable_parameters += list(self.cond_net.parameters())

        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr, weight_decay=1e-5)

    def build_inn(self):
        nodes = [Ff.InputNode(self.horizon)]
        # flatten is unnecessary here??
        # nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))

        if self.cond_net:
            cond = Ff.ConditionNode(self.horizon)
            for k in range(self.no_layer_cond):
                nodes.append(
                    Ff.Node(nodes[-1], Fm.GLOWCouplingBlock, {"subnet_constructor": self.subnet}, conditions=cond))
                nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            for k in range(self.no_layer_without_cond):
                nodes.append(
                    Ff.Node(nodes[-1], Fm.GLOWCouplingBlock, {"subnet_constructor": self.subnet}))
                nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)
        else:
            for k in range(self.no_layer_cond):
                nodes.append(
                    Ff.Node(nodes[-1], Fm.GLOWCouplingBlock, {"subnet_constructor": self.subnet}))
                nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            return Ff.ReversibleGraphNet(nodes + [Ff.OutputNode(nodes[-1])], verbose=False)

    def forward(self, x, c, rev=False):
        if isinstance(x, np.ndarray):
            if isinstance(c, np.ndarray):
                c = self._calculate_condition(torch.from_numpy(c.astype("float32")))
            else:
                c = self._calculate_condition(c)
            z, jac = self.cinn(torch.from_numpy(x.astype("float32")), c=c, rev=rev)
        else:
            c = self._calculate_condition(c)
            z, jac = self.cinn(x.float(), c=c, rev=rev)
        return z, jac

    def _calculate_condition(self, c):
        if c is not None:
            c = self.cond_net(c).reshape((-1, self.horizon))
        return c

    def reverse_sample(self, z, c):
        c = self._calculate_condition(c)
        return self.cinn(z, c=c, rev=True)[0].detach().numpy()


class INNWrapper(BaseEstimator):
    def __init__(self, name: str = "INN", cinn=INN(5e-4, horizon=24, cond_features=168), input="Bldg.124",
                 horizon=24, cond_features=4, epochs=100, val_train_split=0.2,
                 stats_loss=False):
        super().__init__(name)
        self.cinn = cinn
        self.epoch = epochs
        self.horizon = horizon
        self.cond_features = cond_features
        self.val_train_split = val_train_split
        self.is_fitted = False
        self.has_inverse_transform = True
        self.stats_loss = stats_loss

    def get_params(self) -> Dict[str, object]:

        return {
            "epochs": self.epoch,
            "horizon": self.horizon,
            "cond_features": self.cond_features,
        }

    def set_params(self, epochs=None, horizon=None, cond_features=None):
        if epochs is not None:
            self.epoch = epochs
        if horizon is not None:
            self.horizon = horizon
        if cond_features is not None:
            self.cond_features = cond_features

    def save(self, fm: FileManager) -> Dict:
        """
        Saves the modules and the state of the module and returns a dictionary containing the relevant information.

        :param fm: the filemanager which can be used by the module for saving information about the module.
        :type fm: FileManager
        :return: A dictionary containing the information needed for restoring the module
        :rtype:Dict
        """
        json_module = super().save(fm)
        path = fm.get_path(f"module_{self.name}.pickle")
        with open(path, 'wb') as outfile:
            pickle.dump(self.cinn, outfile)
        json_module["module"] = path
        return json_module

    @classmethod
    def load(cls, load_information) -> BaseEstimator:
        """
        Uses the data in load_information for restoring the state of the module.

        :param load_information: The data needed for restoring the state of the module
        :type load_information: Dict
        :return: The restored module
        :rtype: BaseEstimator
        """
        module = super().__class__.load(load_information)
        module.cinn = pickle.load(load_information[f"module_{load_information['name']}"])
        return module

    def fit(self, input_data: xr.DataArray, **kwargs: xr.DataArray):
        # Todo counter does not have to be named with target else pyWATTS tries to add result steps -> for all targets
        start_time = time.time()

        x = input_data.values.reshape((len(input_data), self.horizon))

        rdx = np.random.randint(0, len(input_data) - self.horizon, int(len(input_data) * self.val_train_split))
        conditions_train = []
        conditions_val = []
        counter = {}
        counter_conditions = {}
        x_train = np.delete(x[self.horizon:], rdx, axis=0)
        x_val = x[self.horizon:][rdx]
        cond_names = []
        for key, value in kwargs.items():
            if key.startswith("counter"):
                counter[key.split("_")[-1]] = value.values
            elif key.startswith("con_counter"):
                if key.split("_")[2] in counter_conditions:
                    counter_conditions[key.split("_")[2]][key.split("_")[-1]] = value.values
                else:
                    counter_conditions[key.split("_")[2]] = {key.split("_")[-1]: value.values}
            else:
                cond_names.append(key)
                conditions_train.append(np.delete(value[self.horizon:].values, rdx, axis=0))
                conditions_val.append(value[self.horizon:][rdx].values)

        if len(conditions_train) > 0:
            dataset = TensorDataset(torch.from_numpy(x_train.reshape((len(x_train), self.horizon)).astype("float32")),
                                    torch.from_numpy(np.concatenate(conditions_train, axis=-1).astype("float32")), )
            conditions_val = np.concatenate(conditions_val, axis=-1).astype("float32")

            counter_data = ([], [])
            for k, v in counter.items():
                if not k in counter_conditions:
                    self.logger.warn("Counter data has not the correct shape for the conditional data")
                    continue
                if not all([c in counter_conditions[k] for c in cond_names]):
                    self.logger.warn("Counter data has not the correct shape for the conditional data")
                    continue
                temp_conds = [counter_conditions[k][key] for key in cond_names]
                counter_data[0].append(v)
                counter_data[1].append(np.concatenate(temp_conds, axis=-1))

            if counter_data[0]:
                c_data = np.concatenate(counter_data[0], axis=0)
                c_con = np.concatenate(counter_data[1], axis=0)
                # TODO check shape of c_con, the entries except for the first has to be the same as the original
                dataset_counter = TensorDataset(
                    torch.from_numpy(c_data.reshape((len(c_data), self.horizon)).astype("float32")),
                    torch.from_numpy(c_con.astype("float32")), )
            else:
                dataset_counter = None
        else:
            dataset = TensorDataset(torch.from_numpy(x_train.reshape((len(x_train), self.horizon)).astype("float32")))

            if len(counter) > 0:
                counter_data = np.concatenate(list(map(lambda x: x[0], counter)), axis=0)
                dataset_counter = TensorDataset(
                    torch.from_numpy(counter_data.reshape((len(counter_data), self.horizon)).astype("float32")))
            else:
                dataset_counter = None

        # TODO make batch size a parameter
        data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        nn = (2000000, -1)
        counter = 0
        for epoch in range(self.epoch):
            stop, nn = self._run_epoch(counter, data_loader, epoch, nn, start_time, conditions_val, x_train, x_val,
                                       DataLoader(dataset_counter, 64, shuffle=True) if dataset_counter else None)
            if stop:
                break
        self.logger.info(
            f"--- The entire training of {self.name} takes %s minutes ---" % ((time.time() - start_time) / 60.0))
        self.is_fitted = True

    def _run_epoch(self, counter, data_loader, epoch, nn, start_time, conditions, x_train, x_val, counter_data):
        for i, _input in enumerate(data_loader):
            if len(conditions) > 0:
                (x, c) = _input
            else:
                x = _input[0]
                c = None

            z, log_j = self.cinn(x, c)
            nll = torch.mean(z ** 2) / 2 - torch.mean(log_j) / self.horizon
            nll_stat = 0
            nll_c = 0
            if self.stats_loss:
                noise = np.random.normal(0, 1, size=x.shape)
                synth_stat = np.random.normal(np.random.rand() * 2 - 1, .1, size=c[:,:,-1].shape)
                new_c = c.detach().numpy().copy()
                new_c[:,:,-1] = synth_stat
                new_c = torch.from_numpy(new_c)
                generated = self.cinn.forward(noise, c=new_c, rev=True)[0]
                nll_stat = torch.sqrt(
                    torch.mean((torch.mean(generated, dim=-1) - torch.mean(new_c[:, :, -1]))) ** 2)
               # if nll_stat >= 1:
            #    print("STATS", nll_stat)
                nll = (nll_stat * 0.01 + nll)
            if counter_data is not None:
                counter_input = next(iter(data_loader))
                if len(conditions) > 0:
                    (x, c) = counter_input
                else:
                    x = counter_input[0]
                    c = None
                counter_result = self.cinn(x, c)
                nll_counter = []
                nll_c = 0
                for z_c, log_j_c in list(zip(*counter_result)):
                    l_c = torch.mean(z_c ** 2) / 2 - torch.mean(log_j_c) / self.horizon
                    if l_c.detach().numpy() < -.5:
                        nll_counter.append(l_c)
                if len(nll_counter) > 0:
                    nll_c = sum(nll_counter) / len(nll_counter)

                nll -= nll_c * min(1, abs(nll_c) - 0.5)

            nll.backward()

            # Perhaps first only mle afterwards mle + statsloss
            #   - Perhaps the stats loss pushes it first in the wrong direction?

            torch.nn.utils.clip_grad_norm(self.cinn.trainable_parameters, 1.)
            self.cinn.optimizer.step()
            self.cinn.optimizer.zero_grad()
            if not i % 50:
                counter += 1
                with torch.no_grad():
                    if len(conditions) > 0:
                        z, log_j = self.cinn(x_val, torch.from_numpy(conditions))
                    else:
                        z, log_j = self.cinn(x_val, None)
                    nll_test = torch.mean(z ** 2) / 2 - torch.mean(log_j) / self.horizon
                    print(f"{epoch}, {i}, {len(x_train)}, {nll.item()}, {nll_test.item()}, {nll_stat.item() if nll_stat else 0}, {nll_c.item() if nll_c else 0}")

        return False, nn

    def transform(self, input_data: xr.DataArray, **kwargs: xr.DataArray) -> xr.DataArray:
        x = input_data.values.reshape((len(input_data), self.horizon))
        c = self._get_conditions(kwargs)

        return numpy_to_xarray(self.cinn.forward(x, c=c)[0].detach().numpy(), input_data, self.name)

    def inverse_transform(self, input_data: xr.DataArray, **kwargs: xr.DataArray) -> xr.DataArray:

        x = input_data.values.reshape((len(input_data), self.horizon))
        x = torch.from_numpy(x.astype("float32"))
        c = self._get_conditions(kwargs)
        return numpy_to_xarray(self.cinn.reverse_sample(x, c=c).reshape((-1, self.horizon)), input_data,
                               f"reverse_{self.name}")

    def _get_conditions(self, kwargs):
        conditions = []
        for key, value in kwargs.items():
            if not (key.startswith("counter") or key.startswith("con_counter")):
                conditions.append(value.values)
        if not conditions:
            c = None
        else:
            c = torch.Tensor(np.concatenate(conditions, axis=-1))
        return c

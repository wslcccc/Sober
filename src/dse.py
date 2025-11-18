from src.saver import saver
from src.utils import MLP, load, get_save_path, argsort, get_root_path, get_src_path, \
    _get_y_with_target, _get_y
from src.programl_data import print_data_stats, _check_any_in_str, NON_OPT_PRAGMAS, WITH_VAR_PRAGMAS, \
    _in_between, _encode_edge_dict, _encode_edge_torch, _encode_X_torch, create_edge_index
from src.model import Net
from src.parameter import DesignSpace, DesignPoint, DesignParameter, get_default_point, topo_sort_param_ids, \
    compile_design_space, gen_key_from_design_point
from src.config_ds import build_config
from src.result import Result
from CoGNN.model_parse import GumbelArgs, EnvArgs, ActionNetArgs, ActivationType
import json
import os
from math import ceil, inf, exp, log10
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Deque, Dict, List, Optional, Set, Union, Generator, Any
import sys
import copy
import itertools
import networkx as nx
from collections import OrderedDict
from glob import glob
import pickle
from torch.nn import Sequential, Linear, ReLU
from typing import NamedTuple, Any, Callable
from random import uniform, randint, random
import numpy as np
import time
from typing import Dict, List, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from result import Result
from parameter import gen_key_from_design_point
import pickle
from config import FLAGS
from os.path import join
import numpy

SAVE_DIR = join(get_root_path(), f'save_models_and_data')
SAVE_DIR_CLASS = join(get_root_path(), f'save_models_and_data')
def gin_mlp_func() -> Callable:
    def mlp_func(in_channels: int, out_channels: int, bias: bool):
        return Sequential(Linear(in_channels, out_channels, bias=bias),
                ReLU(), Linear(out_channels, out_channels, bias=bias))
    return mlp_func

out_dim = FLAGS.out_dim
gin_mlp_func = gin_mlp_func()

gumbel_args = GumbelArgs(learn_temp=FLAGS.learn_temp, temp_model_type=FLAGS.temp_model_type, tau0=FLAGS.tau0,
                                 temp=FLAGS.temp, gin_mlp_func=gin_mlp_func)
env_args = \
EnvArgs(model_type=FLAGS.env_model_type, num_layers=FLAGS.env_num_layers, env_dim=FLAGS.env_dim,
        layer_norm=FLAGS.layer_norm, skip=FLAGS.skip, batch_norm=FLAGS.batch_norm, dropout=FLAGS.dropout,
        in_dim=FLAGS.num_features , out_dim=FLAGS.D, dec_num_layers=FLAGS.dec_num_layers, gin_mlp_func=gin_mlp_func,
        act_type=ActivationType.RELU)
action_args = \
        ActionNetArgs(model_type=FLAGS.act_model_type, num_layers=FLAGS.act_num_layers,
        hidden_dim=FLAGS.act_dim, dropout=FLAGS.dropout, act_type=ActivationType.RELU,
        gin_mlp_func=gin_mlp_func, env_dim=FLAGS.env_dim)

class GNNModel():
    def __init__(self, path, saver, multi_target=True, task='regression', num_layers=FLAGS.num_layers, D=FLAGS.D,
                 target=FLAGS.target, model_name=f'{FLAGS.model_tag}_model_state_dict.pth', encoder_name='encoders'):
        """
        >>> self.encoder.keys()
        dict_keys(['enc_ntype', 'enc_ptype', 'enc_itype', 'enc_ftype', 'enc_btype', 'enc_ftype_edge', 'enc_ptype_edge'])

        """
        model_name = f'{task}_model_state_dict.pth'
        self.log = saver
        self.path = path
        if task == 'regression':
            if FLAGS.model_path == None:
                self.model_path = join(self.path, model_name)
            else:
                self.model_path = FLAGS.model_path
        else:
            if FLAGS.class_model_path == None:
                self.model_path = join(self.path, model_name)
            else:
                self.model_path = FLAGS.class_model_path
        if FLAGS.encoder_path == None:
            self.encoder_path = join(self.path, encoder_name)
        else:
            self.encoder_path = FLAGS.encoder_path
        self.num_features = FLAGS.num_features  # 153
        self.model = Net(gumbel_args=gumbel_args, env_args=env_args, action_args=action_args).to(
            FLAGS.device)
        self.model.load_state_dict(torch.load(join(self.model_path), map_location=torch.device('cpu')))
        saver.info(f'loaded {self.model_path}')
        self.encoder = load(self.encoder_path)

    def encode_node(self, g, point: DesignPoint):  # prograML graph
        X_ntype = []  # node type <attribute id="3" title="type" type="long" />
        X_ptype = []  # pragma type
        X_numeric = []
        X_itype = []  # instruction type (text) <attribute id="2" title="text" type="string" />
        X_ftype = []  # function type <attribute id="1" title="function" type="long" />
        X_btype = []  # block type <attribute id="0" title="block" type="long" />

        for node, ndata in g.nodes(data=True):  # TODO: node ordering
            numeric = 0
            if 'full_text' in ndata and 'pragma' in ndata['full_text']:
                # print(ndata['content'])
                p_text = ndata['full_text'].rstrip()
                assert p_text[0:8] == '#pragma '
                p_text_type = p_text[8:].upper()

                if _check_any_in_str(NON_OPT_PRAGMAS, p_text_type):
                    p_text_type = 'None'
                else:
                    if _check_any_in_str(WITH_VAR_PRAGMAS, p_text_type):
                        # HLS DEPENDENCE VARIABLE=CSIYIY ARRAY INTER FALSE
                        # HLS DEPENDENCE VARIABLE=<> ARRAY INTER FALSE
                        t_li = p_text_type.split(' ')
                        for i in range(len(t_li)):
                            if 'VARIABLE=' in t_li[i]:
                                t_li[i] = 'VARIABLE=<>'
                            elif 'DEPTH=' in t_li[i]:
                                t_li[i] = 'DEPTH=<>'  # TODO: later add back
                            elif 'DIM=' in t_li[i]:
                                numeric = int(t_li[i][4:])
                                t_li[i] = 'DIM=<>'
                            elif 'LATENCY=' in t_li[i]:
                                numeric = int(t_li[i][8:])
                                t_li[i] = 'LATENCY=<>'
                        p_text_type = ' '.join(t_li)

                    if point is not None:
                        t_li = p_text_type.split(' ')
                        for i in range(len(t_li)):
                            if 'AUTO{' in t_li[i]:
                                # print(t_li[i])
                                auto_what = _in_between(t_li[i], '{', '}')
                                numeric = point[auto_what]
                                if type(numeric) is not int:
                                    t_li[i] = numeric
                                    numeric = 0  # TODO: ? '', 'off', 'flatten'
                                else:
                                    t_li[i] = 'AUTO{<>}'
                                break
                        p_text_type = ' '.join(t_li)
                    else:
                        assert 'AUTO' not in p_text_type
                    # t = ' '.join(t.split(' ')[0:2])
                # print('@@@@@', t)
                ptype = p_text_type
            else:
                ptype = 'None'

            X_ntype.append([ndata['type']])
            X_ptype.append([ptype])
            X_numeric.append([numeric])
            X_itype.append([ndata['text']])
            X_ftype.append([ndata['function']])
            X_btype.append([ndata['block']])

        node_dict = {'X_ntype': X_ntype, 'X_ptype': X_ptype,
                     'X_numeric': X_numeric, 'X_itype': X_itype,
                     'X_ftype': X_ftype, 'X_btype': X_btype}

        enc_ntype = self.encoder['enc_ntype']
        enc_ptype = self.encoder['enc_ptype']
        enc_itype = self.encoder['enc_itype']
        enc_ftype = self.encoder['enc_ftype']
        enc_btype = self.encoder['enc_btype']

        return _encode_X_torch(node_dict, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)

    def encode_edge(self, g):
        edge_dict = _encode_edge_dict(g)
        enc_ptype_edge = self.encoder['enc_ptype_edge']
        enc_ftype_edge = self.encoder['enc_ftype_edge']

        return _encode_edge_torch(edge_dict, enc_ftype_edge, enc_ptype_edge)

    def perf_as_quality(self, new_result: Result) -> float:
        perf = math.pow(2, 2 * new_result.perf)
        new_result.perf = perf / FLAGS.normalizer
        return new_result.perf

    def quantify_util(self, result: Result) -> float:
        utils = [
            5 * ceil(u * 100 / 5) / 100 + FLAGS.epsilon for k, u in result.res_util.items()
            if k.startswith('util')
        ]

        # Compute the area
        return sum(utils) / 4

    def eff_as_quality(self, new_result: Result) -> float:
        """Compute the quality of the point by resource efficiency.

        Args:
            new_result: The new result to be qualified.
            ref_result: The reference result.

        Returns:
            The quality value (negative finite differnece). Larger the better.
        """
        area = sum([0.25 * u for k, u in new_result.res_util.items() if k.startswith('util')])
        return log10(abs(1 / (new_result.perf * area)) + 1)

    def test(self, loader, config, mode='regression'):
        self.model.eval()

        i = 0
        results: List[Result] = []
        target_list = FLAGS.target
        if not isinstance(FLAGS.target, list):
            target_list = [FLAGS.target]
        for data in loader:
            data = data.to(FLAGS.device)
            out_dict, loss, loss_dict = self.model(data)
            if mode == 'regression':
                for i in range(len(out_dict['perf'])):
                    curr_result = Result()
                    curr_result.point = data[i].point
                    for target_name in target_list:
                        out = out_dict[target_name]
                        out_value = out[i].item()
                        if target_name == 'perf':
                            curr_result.perf = out_value
                            if FLAGS.encode_log:
                                curr_result.actual_perf = 2 ** out_value
                            else:
                                curr_result.actual_perf = out_value
                        elif target_name in curr_result.res_util.keys():
                            if out_value < 0:
                                curr_result.res_util[target_name] = 0
                            else:
                                curr_result.res_util[target_name] = out_value
                        else:
                            raise NotImplementedError()
                    quality = self.perf_as_quality(curr_result)
                    curr_result.area = self.quantify_util(curr_result)
                    curr_result.quality = quality
                    results.append(curr_result)
            elif mode == 'class':
                _, pred = torch.max(out_dict['perf'], 1)
                labels = _get_y_with_target(data, 'perf')
                # saver.debug(f'pred: {pred}, labels: {labels}')
                return (pred == labels)
            else:
                raise NotImplementedError()

        return results

class Explorer():
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool = True, prune_invalid=False):
        """Constructor.

        Args:
            ds: DesignSpace
        """
        self.run_dse = True
        self.log = saver
        self.kernel_name = kernel_name
        self.config_path = join(path_kernel, f'{kernel_name}_ds_config.json')
        self.config = self.load_config()
        self.timeout = 60 * 60
        self.ds, self.ds_size = compile_design_space(
            self.config['design-space']['definition'],
            None,
            self.log)
        self.batch_size = 1
        # Status checking
        self.num_top_designs = 2
        self.key_perf_dict = OrderedDict()
        self.best_results_dict = {}
        self.best_result: Result = Result()
        self.explored_point = 0
        self.ordered_pids = self.topo_sort_param_ids(self.ds)
        self.GNNmodel = GNNModel(SAVE_DIR, self.log, multi_target=True, task='regression', num_layers=FLAGS.num_layers,
                                 D=FLAGS.D)
        self.best_save_results = {}
        if FLAGS.separate_perf:
            perf_target = ['perf', 'util-LUT', 'util-FF', 'util-DSP']
            self.GNNmodel_perf = GNNModel(SAVE_DIR, self.log, multi_target=True, task='regression_perf', num_layers=8,
                                          D=64, target=perf_target)
        gexf_file = []
        for f in glob(path_graph + "/*"):
            if f.endswith('.gexf') and kernel_name in f:
                if f[len(path_graph) + len(kernel_name) + 1] == '_':
                    gexf_file.append(f)

        # gexf_file = sorted([f for f in glob(path_graph + "/*") if f.endswith('.gexf') and kernel_name in f])
        # print(gexf_file, glob(path_graph))
        assert len(gexf_file) >= 1
        # self.graph_path = join(path_graph, f'{kernel_name}_processed_result.gexf')

        self.graph_path = join(path_graph, gexf_file[0])
        self.graph = nx.read_gexf(self.graph_path)
        self.prune_invalid = prune_invalid
        if self.prune_invalid:
            self.GNNmodel_valid = GNNModel(SAVE_DIR_CLASS, self.log, multi_target=False, task='class',
                                           num_layers=FLAGS.num_layers, D=FLAGS.D)
        if self.ds_size <= 100:
            self.result_number = 10
            self.stop_cond = 100
        elif 100 < self.ds_size <= 10000:
            self.result_number = 20
            self.stop_cond = 2000
        elif 10000 < self.ds_size <= 100000:
            self.result_number = 30
            self.stop_cond = 3000
        elif 100000 < self.ds_size <= 1e6:
            self.result_number = 30
            self.stop_cond = 4000
        elif 1e6 < self.ds_size <= 1e7:
            self.result_number = 30
            self.stop_cond = 5000
        else:
            self.result_number = 60
            self.stop_cond = 6000

    def topo_sort_param_ids(self, space: DesignSpace) -> List[str]:
        return topo_sort_param_ids(space)

    def load_config(self) -> Dict[str, Any]:
        """Load the DSE configurations.

        Returns:
            A dictionary of configurations.
        """

        try:
            if not os.path.exists(self.config_path):
                self.log.error(('Config JSON file not found: %s', self.config_path))
                raise RuntimeError()

            self.log.info('Loading configurations')
            with open(self.config_path, 'r', errors='replace') as filep:
                try:
                    user_config = json.load(filep)
                except ValueError as err:
                    self.log.error(('Failed to load config: %s', str(err)))
                    raise RuntimeError()

            config = build_config(user_config, self.log)
            if config is None:
                self.log.error(('Config %s is invalid', self.config_path))
                raise RuntimeError()
        except RuntimeError:
            sys.exit(1)

        return config

    def apply_design_point(self, g, point: DesignPoint, mode='regression') -> Data:
        X = self.GNNmodel.encode_node(g, point)
        edge_attr = self.GNNmodel.encode_edge(g)
        edge_index = create_edge_index(g)

        d_node = dict()
        resources = ['BRAM', 'DSP', 'LUT', 'FF']
        keys = ['perf', 'actual_perf', 'quality']
        for r in resources:
            keys.append('util-' + r)
            keys.append('total-' + r)
        for key in keys:
            d_node[key] = 0
        if mode == 'class':  ## default: point is valid
            d_node['perf'] = 1

        if 'regression' in mode:
            data = Data(
                x=X,
                edge_index=edge_index,
                perf=d_node['perf'],
                actual_perf=d_node['actual_perf'],
                quality=d_node['quality'],
                util_BRAM=d_node['util-BRAM'],
                util_DSP=d_node['util-DSP'],
                util_LUT=d_node['util-LUT'],
                util_FF=d_node['util-FF'],
                total_BRAM=d_node['total-BRAM'],
                total_DSP=d_node['total-DSP'],
                total_LUT=d_node['total-LUT'],
                total_FF=d_node['total-FF'],
                point=point,
                edge_attr=edge_attr
            )
        elif mode == 'class':
            data = Data(
                x=X,
                edge_index=edge_index,
                perf=d_node['perf'],
                edge_attr=edge_attr,
                kernel=self.kernel_name
            )
        else:
            raise NotImplementedError()

        return data

    def update_best(self, result: Result):
        """Keep tracking the best result found in this explorer.

        Args:
            result: The new result to be checked.

        """
        # if result.valid and result.quality > self.best_result.quality:
        update_flag = False
        REF = min
        if self.key_perf_dict:
            key_refs_perf = REF(self.key_perf_dict, key=(lambda key: self.key_perf_dict[key]))
            refs_perf = self.key_perf_dict[key_refs_perf]
        else:
            if REF == min:
                refs_perf = float(-inf)
            else:
                refs_perf = float(inf)
        point_key = gen_key_from_design_point(result.point)
        if point_key not in self.key_perf_dict and result.valid and REF(result.quality,
                                                                        refs_perf) != result.quality:  # if the new result is better than the references designs
            self.best_result = result
            self.log.info(('Found a better result at {}: Quality {:.1e}, Perf {:.1e}'.format(
                self.explored_point, result.quality, result.perf)))
            if len(self.key_perf_dict.keys()) >= self.num_top_designs:
                # replace maxmimum performance value
                key_refs_perf = REF(self.key_perf_dict, key=(lambda key: self.key_perf_dict[key]))
                self.best_results_dict.pop((self.key_perf_dict[key_refs_perf], key_refs_perf))
                self.key_perf_dict.pop(key_refs_perf)
            attrs = vars(result)
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
            self.key_perf_dict[point_key] = result.quality
            self.best_results_dict[(result.quality, point_key)] = result
            update_flag = True
        return update_flag
    def gen_options(self, point: DesignPoint, pid: str, default=False) -> List[Union[int, str]]:
        """Evaluate available options of the target design parameter.

        Args:
            point: The current design point.
            pid: The target design parameter ID.

        Returns:
            A list of available options.
        """
        if default:
            dep_values = {dep: point[dep].default for dep in self.ds[pid].deps}
        else:
            dep_values = {dep: point[dep] for dep in self.ds[pid].deps}
        dep_values = {dep: point[dep] for dep in self.ds[pid].deps}
        options = eval(self.ds[pid].option_expr, dep_values)
        if '' in options:
            options.pop(options.index(''))
        if options is None:
            self.log.error(f'Failed to evaluate {self.ds[pid].option_expr} with dep {str(dep_values)}')
            print('Error: failed to manipulate design points')
            sys.exit(1)

        return options

    def get_order(self, point: DesignPoint, pid: str) -> int:
        """Evaluate the order of the current value.

        Args:
            point: The current design point.
            pid: The target design parameter ID.

        Returns:
            The order.
        """

        if not self.ds[pid].order:
            return 0

        order = eval(self.ds[pid].order['expr'], {self.ds[pid].order['var']: point[pid]})
        if order is None or not isinstance(order, int):
            self.log.warning(f'Failed to evaluate the order of {pid} with value {str(point[pid])}: {str(order)}')
            return 0

        return order

    def update_child(self, point: DesignPoint, pid: str) -> None:
        """Check values of affect parameters and update them in place if it is invalid.

        Args:
            point: The current design point.
            pid: The design parameter ID that just be changed.
        """

        pendings = [child for child in self.ds[pid].child if self.validate_value(point, child)]
        for child in pendings:
            self.update_child(point, child)

    def validate_point(self, point: DesignPoint) -> bool:
        """Check if the current point is valid and set it to the closest value if not.

        Args:
            point: The current design point.
            pid: The design parameter ID that just be changed.

        Returns:
            True if the value is changed.
        """

        changed = False
        for pid in point.keys():
            options = self.gen_options(point, pid)
            value = point[pid]
            if not options:  # All invalid (something not right), set to default
                self.log.warning(f'No valid options for {pid} with point {str(point)}')
                point[pid] = self.ds[pid].default
                changed = True
                continue

            if isinstance(value, int):
                # Note that we assume all options have the same type (int or str)
                cand = min(options, key=lambda x: abs(int(x) - int(value)))
                if cand != value:
                    point[pid] = cand
                    changed = True
                    continue

            if value not in options:
                point[pid] = self.ds[pid].default
                changed = True
                continue

        return changed

    def validate_value(self, point: DesignPoint, pid: str) -> bool:
        """Check if the current value is valid and set it to the closest value if not.

        Args:
            point: The current design point.
            pid: The design parameter ID that just be changed.

        Returns:
            True if the value is changed.
        """

        options = self.gen_options(point, pid)
        value = point[pid]
        if not options:  # All invalid (something not right), set to default
            self.log.warning(f'No valid options for {pid} with point {str(point)}')
            point[pid] = self.ds[pid].default
            return False

        if isinstance(value, int):
            # Note that we assume all options have the same type (int or str)
            cand = min(options, key=lambda x: abs(int(x) - int(value)))
            if cand != value:
                point[pid] = cand
                return True

        if value not in options:
            point[pid] = self.ds[pid].default
            return True
        return False

    def move_by(self, point: DesignPoint, pid: str, step: int = 1) -> int:
        """Move N steps of pid parameter's value in a design point in place.

        Args:
            point: The design point to be manipulated.
            pid: The target design parameter.
            step: The steps to move. Note that step can be positive or negatie,
                  but we will not move cirulatory even the step is too large.

        Returns:
            The actual move steps.
        """

        try:
            options = self.gen_options(point, pid)
            idx = options.index(point[pid])
        except (AttributeError, ValueError) as err:
            self.log.error(
                f'Fail to identify the index of value {point[pid]} of parameter {pid} at design point {str(point)}: {str(err)}')
            print('Error: failed to manipulate design points')
            sys.exit(1)

        target = idx + step
        if target >= len(options):
            target = len(options) - 1
        elif target < 0:
            target = 0

        if target != idx:
            point[pid] = options[target]
            self.update_child(point, pid)
        return target - idx
    def traverse(self, point: DesignPoint, idx: int) -> Generator[DesignPoint, None, None]:
        """DFS traverse the design space and yield leaf points.

        Args:
            point: The current design point.
            idx: The current manipulated parameter index.

        Returns:
            A resursive generator for traversing.
        """

        if idx == len(self.ordered_pids):
            # Finish a point
            yield point
        else:
            yield from self.traverse(point, idx + 1)

            # Manipulate idx-th point
            new_point = self.clone_point(point)
            while self.move_by(new_point, self.ordered_pids[idx]) == 1:
                yield from self.traverse(new_point, idx + 1)
                new_point = self.clone_point(new_point)

    @staticmethod
    def clone_point(point: DesignPoint) -> DesignPoint:
        return dict(point)

    def get_results(self, population: List[DesignPoint]) -> List[Result]:
        data_list = []
        for point in population:
            data_list.append(self.apply_design_point(self.graph, point))

        test_loader = DataLoader(data_list, batch_size=self.batch_size)  # TODO
        results = self.GNNmodel.test(test_loader, self.config['evaluate'], mode='regression')
        return results

    # Large language model-evolutionary computation part
    def get_config_dafault_options(self):

        defaults_dict = {key: self.ds[key].default for key in self.ordered_pids}
        config_options = {}
        config_cond = {}
        for key in self.ordered_pids:
            if self.ds[key].deps:
                value = self.ds[key].option_expr.split('if')
                config_options[key] = eval(value[0] + ']')
                config_cond[key] = value[1][:-1]
            else:
                config_options[key] = eval(self.ds[key].option_expr)
                config_cond[key] = ''
        for i, j in config_options.items():  # 返回键值对
            if '' in j:
                j.remove('')
        return defaults_dict, config_options, config_cond


    def transfer_res_to_config(self, res, logger, co):
        logger.info('starting to transfer res to config')
        res = res.split('\n')
        res = [i for i in res if i.find('[') >= 0 and i.find(']') >= 0]
        k = list(co.keys())
        v = list(co.values())
        res_1 = []
        if len(res) < 2:
            return []
        for r in res:
            flag = False
            d1 = {}
            # r_1 = eval((('[' + r.split('[')[1]).split(']')[0]) + ']')
            r_1 = eval((r[r.find('['):r.find(']') + 1]))
            if len(r_1) < len(k):
                continue
            for nums, i in enumerate(k):
                if r_1[nums] in v[nums]:
                    d1[i] = r_1[nums]
                else:
                    flag = True
                    break
            if flag:
                continue
            res_1.append(d1)
        return res_1

    def generate_all_solutions(self, default_dict, pragmas_possible_values, config_cond):
        # Extract the order of pragmas from default_dict
        pragma_order = list(default_dict.keys())
        # Get the possible values for each pragma in the correct order
        value_lists = [pragmas_possible_values[pragma] for pragma in pragma_order]
        # Generate all possible combinations of values
        all_combinations = itertools.product(*value_lists)
        solutions = []

        for combination in all_combinations:
            # Create a configuration dictionary from the combination
            config = dict(zip(pragma_order, combination))
            is_valid = True
            temp_dict = config
            for key in temp_dict.keys():
                if config_cond[key] != '':
                    cond = config_cond[key]
                    dep_list = self.ds[key].deps
                    x = temp_dict[key]
                    temp = cond
                    for dep in dep_list:
                        if type(temp_dict[dep]) == int:
                            temp = temp.replace(dep, str(temp_dict[dep]))
                        else:
                            temp = temp.replace(dep, f'\'{temp_dict[dep]}\'')
                    if eval(temp) == False:
                        is_valid = False
                        break
            if is_valid:
                solutions.append(config)

        return solutions



    def run(self) -> None:
        """The main function of the explorer to launch the search algorithm.

        Args:
            algo_name: The corresponding algorithm name for running this exploration.
            algo_config: The configurable values for the algorithm.
        """
        raise NotImplementedError()

import random
class NSGAIIExplorer(Explorer):
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool = True,
                 prune_invalid=FLAGS.prune_class, point: DesignPoint = None):
        """Constructor.

        Args:
            ds: Design space.
        """
        super(NSGAIIExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.batch_size = 1
        self.log.info('Done init')
        start_time = time.time()
        if self.run_dse:
            self.run()
            attrs = vars(self.best_result)
            self.log.info('Best Results Found:')
            i = 1
            with open(join('/home/test/xulei/sober/Best_result/NSGA-II/test', f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
        end_time = time.time()
        print(f'runtime: {end_time - start_time}')
    def run(self) -> None:
        defaults_dict, config_options, config_cond = self.get_config_dafault_options()
        # init population
        population = []
        for i in range(self.result_number):
            init_solution = {}
            for key, value in config_options.items():
                init_solution[key] = value[randint(0, len(value) - 1)]
            population.append(init_solution)
        p_cs = 0.1
        p_mt = 0.1
        while self.explored_point <= self.stop_cond:
            # evaluate solution
            print('------------------------------------------------------')
            print(f'explored point {self.explored_point}/{self.stop_cond}')
            print('------------------------------------------------------')
            results = self.get_results(population)
            for r in results:
                self.explored_point += 1
                fitness = [i.quality for i in self.best_results_dict.values()]
                self.best_save_results[self.explored_point] = fitness
                if isinstance(r, Result):
                    attrs = vars(r)
                    self.log.debug(f'Evaluating Design')
                    self.log.debug(', '.join("%s: %s" % item for item in attrs.items()))
                    flag = self.update_best(r)
            # generate population
            # selection
            fitness_1 = [i.quality for i in results]
            selected = []
            for _ in range(self.result_number):
                contestants = random.sample(population, k=3)
                inx = [population.index(i) for i in contestants]
                val_inx = [fitness_1[i] for i in inx]
                max_inx = fitness_1.index(max(val_inx))
                selected.append(population[max_inx])
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 >= len(selected):
                    break
                c1, c2 = {}, {}
                p1, p2 = selected[i], selected[i+1]
                for para in config_options.keys():
                    if random.random() < p_cs:
                        c1[para] = p2[para]
                        c2[para] = p1[para]
                    else:
                        c1[para] = p1[para]
                        c2[para] = p2[para]
                offspring.append(c1)
                offspring.append(c2)
            temp = {}
            population_1 = []
            for i in offspring:
                temp = i.copy()
                for j in config_options.keys():
                    if random.random() < p_mt:
                        temp[j] = config_options[j][randint(0, len(config_options[j]) - 1)]
                population_1.append(temp)
            population = population_1
        self.log.info(f'Explored {self.explored_point} points')

class SAExplorer(Explorer):
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool = True,
                 prune_invalid=FLAGS.prune_class, point: DesignPoint = None):
        """Constructor.

        Args:
            ds: Design space.
        """
        super(SAExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.batch_size = 1
        self.log.info('Done init')
        start_time = time.time()
        if self.run_dse:
            self.run()
            attrs = vars(self.best_result)
            self.log.info('Best Results Found:')
            i = 1
            with open(join('/home/test/xulei/sober/Best_result/SA/test', f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
        end_time = time.time()
        print(f'runtime: {end_time - start_time}')

    def run(self) -> None:
        defaults_dict, config_options, config_cond = self.get_config_dafault_options()
        # get total solutions
        temperature_1 = FLAGS.initial_temperature
        # init solution
        cand_solutions = []
        init_solution = {}
        for i in range(self.result_number):
            for key, value in config_options.items():
                init_solution[key] = value[randint(0, len(value) - 1)]
            cand_solutions.append(init_solution)
        config_len = {key:len(value) for key, value in config_options.items()}
        neighbor_dis = [ceil(value * FLAGS.neighbor_distance_rate) for value in config_len.values()]
        while self.explored_point <= self.stop_cond and temperature_1 >= FLAGS.stop_temperature:
            # evaluate solution
            print('------------------------------------------------------')
            print(f'explored point {self.explored_point}/{self.stop_cond}')
            print('------------------------------------------------------')
            results = self.get_results(cand_solutions)
            for r in results:
                self.explored_point += 1
                fitness = [i.quality for i in self.best_results_dict.values()]
                self.best_save_results[self.explored_point] = fitness
                if isinstance(r, Result):
                    attrs = vars(r)
                    self.log.debug(f'Evaluating Design')
                    self.log.debug(', '.join("%s: %s" % item for item in attrs.items()))
                    flag = self.update_best(r)
            solution_list = [i.point for i in self.best_results_dict.values()]
            cand_values = [list(i)[0] for i in self.best_results_dict.keys()]
            best_solutions = []
            for i in solution_list:
                for j in i.keys():
                    temp = i[j]
                    if torch.is_tensor(temp):
                        i[j] = int(temp)
                best_solutions.append(i)
            temp_solution = copy.deepcopy(best_solutions)
            cand_solutions = []
            for i in range(self.result_number):
                for num, solution in enumerate(temp_solution):
                    new_solution = {}
                    for nums, (key, value) in enumerate(solution.items()):
                        config_inx = config_options[key]
                        inx = config_inx.index(value)
                        df = neighbor_dis[nums]
                        dis = randint(-df, df)
                        inx += dis
                        if 0 <= inx <= len(config_inx) - 1:
                            new_solution[key] = config_inx[inx]
                        else:
                            new_solution[key] = value
                    cur_val = cand_values[num]
                    new_val = self.get_results([new_solution])[0].quality
                    delta_val = cur_val - new_val
                    if delta_val < 0 or random.random() < exp(-delta_val/ temperature_1):
                        cand_solutions.append(new_solution)
                    else:
                        cand_solutions += temp_solution
            temperature_1 = temperature_1 / (1 + FLAGS.cooling_rate)
        self.log.info(f'Explored {self.explored_point} points')



class ACOExplorer(Explorer):
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool = True,
                 prune_invalid=FLAGS.prune_class, point: DesignPoint = None):
        """Constructor.

        Args:
            ds: Design space.
        """
        super(ACOExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.defaults_dict, self.config_options, self.config_cond = self.get_config_dafault_options()
        self.batch_size = 1
        self.params = self.config_options
        self.param_names = list(self.config_options.keys())
        self.n_params = len(self.param_names)
        self.alpha = 1.0
        self.beta = 2.0
        self.rho = 0.01

        self.pheromone = {param: np.ones(len(values)) for param, values in self.params.items()}

        self.pareto_front = []
        self.log.info('Done init')
        start_time = time.time()
        if self.run_dse:
            self.run()
            attrs = vars(self.best_result)
            self.log.info('Best Results Found:')
            i = 1
            with open(join('/home/test/xulei/sober/Best_result/ACO/test', f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
        end_time = time.time()
        print(f'runtime: {end_time - start_time}')

    class Ant:
        def __init__(self, aco):
            self.aco = aco
            self.solution = {}
            self.fitness = 0
            self.violation = 0

        def construct_solution(self):
            if self.aco.explored_point == 0 and self.aco.explored_point % 200 == 0:
                init_solution = {}
                for key, value in self.aco.config_options.items():
                    init_solution[key] = value[randint(0, len(value) - 1)]
                self.solution = init_solution

            else:
                for key in self.aco.params.keys():
                    prob = self.aco.calculate_probability(key)
                    selected_idx = np.random.choice(len(prob), p=prob)
                    self.solution[key] = self.aco.params[key][selected_idx]

    def run(self) -> None:
        while self.explored_point <= self.stop_cond:
            print('------------------------------------------------------')
            print(f'explored point {self.explored_point}/{self.stop_cond}')
            print('------------------------------------------------------')
            ants = [self.Ant(self) for _ in range(self.result_number)]
            solutions = []
            for ant in ants:
                ant.construct_solution()
                result = self.get_results([ant.solution])
                for i in result:
                    self.explored_point += 1
                    self.best_save_results[self.explored_point] = [i.quality for i in self.best_results_dict.values()]
                    self.update_best(i)
                ant.fitness = result[0].quality
            self.update_pareto_front(ants)
            self.update_pheromone(ants)
        self.log.info(f'Explored {self.explored_point} points')

    def calculate_probability(self, param):
        value = self.params[param]
        pheromone = self.pheromone[param]
        if type(value[0]) == str:
            values = [i + 1 for i in range(len(value))]
        else:
            values = value
        heuristic = np.array([1.0 / (v + 1e-6) for v in values])
        probabilities = (pheromone ** self.alpha) * (heuristic ** self.beta)
        probabilities /= np.sum(probabilities)
        return probabilities

    def update_pheromone(self, ants: List[Ant]):
        for param in self.param_names:
            self.pheromone[param] *= (1 - self.rho)

        for ant in self.pareto_front:
            for param in self.param_names:
                idx = self.params[param].index(ant.solution[param])
                self.pheromone[param][idx] += 0.5

    def update_pareto_front(self, ants: List[Ant]):
        for ant in ants:
            if ant.violation > 0:
                continue
            is_pareto = True
            for front_sol in self.pareto_front[:]:
                if self.is_dominated(ant.fitness, front_sol.fitness):
                    self.pareto_front.remove(front_sol)
                elif self.is_dominated(front_sol.fitness, ant.fitness):
                    is_pareto = False
                    break
            if is_pareto:
                self.pareto_front.append(ant)

    def is_dominated(self, sol_a, sol_b) -> bool:
        better = False
        if sol_a > sol_b:
            return True
        else:
            return better

class ExhaustiveExplorer(Explorer):
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool = True,
                 prune_invalid=FLAGS.prune_class, point: DesignPoint = None):
        """Constructor.

        Args:
            ds: Design space.
        """
        super(ExhaustiveExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.batch_size = 1
        self.log.info('Done init')

        if self.run_dse:
            self.run()
            attrs = vars(self.best_result)
            self.log.info('Best Results Found:')
            i = 1
            with open(join('/home/test/xulei/sober/Best_result/ref', f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
                i += 1

    def gen(self) -> Generator[List[DesignPoint], Optional[Dict[str, Result]], None]:
        # pylint:disable=missing-docstring

        self.log.info('Launch exhaustive search algorithm')

        traverser = self.traverse(get_default_point(self.ds), 0)
        iter_cnt = 0
        while True:
            next_points: List[DesignPoint] = []
            try:
                iter_cnt += 1
                self.log.debug(f'Iteration {iter_cnt}')
                while len(next_points) < self.batch_size:
                    next_points.append(next(traverser))
                    self.log.debug(f'Next point: {str(next_points[-1])}')
                yield next_points
            except StopIteration:
                if next_points:
                    yield next_points
                break

        self.log.info('No more points to be explored, stop.')

    def run(self) -> None:
        # pylint:disable=missing-docstring

        # Create a search algorithm generator
        gen_next = self.gen()

        timer = time.time()
        duplicated_iters = 0
        while (time.time() - timer) <= self.timeout:
            try:
                # Generate the next set of design points
                next_points = next(gen_next)
                self.log.debug(f'The algorithm generates {len(next_points)} design points')
            except StopIteration:
                break

            results = self.get_results(next_points)
            for r in results:
                if isinstance(r, Result):
                    attrs = vars(r)
                    self.log.debug(f'Evaluating Design')
                    self.log.debug(', '.join("%s: %s" % item for item in attrs.items()))
                    self.update_best(r)
            self.explored_point += len(results)

class ActorCriticNetwork(nn.Module):

    def __init__(self, state_dim: int, action_dims: List[int], device: str = 'cuda:0'):
        super().__init__()
        self.device = device
        self.action_dims = action_dims

        self.common_fc1 = nn.Linear(state_dim, 256)
        self.common_fc2 = nn.Linear(256, 256)
        self.common_fc3 = nn.Linear(256, 128)

        self.actor_heads = nn.ModuleList([
            nn.Linear(128, dim) for dim in action_dims
        ])

        self.critic = nn.Linear(128, 1)

        self.optimizer = Adam(self.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.9
        )

        self.to(device)

    def forward(self, state):

        x = F.leaky_relu(self.common_fc1(state), 0.3)
        x = F.leaky_relu(self.common_fc2(x), 0.1)
        x = torch.tanh(self.common_fc3(x))

        action_probs = []
        for head in self.actor_heads:
            probs = F.softmax(head(x), dim=-1)
            action_probs.append(probs)

        value = self.critic(x)

        return action_probs, value


class ACExplorer(Explorer):

    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str,
                 run_dse: bool = True, prune_invalid: bool = False,
                 point: Optional[Dict[str, Any]] = None):

        super().__init__(path_kernel, kernel_name, path_graph, False, prune_invalid)

        self.device = FLAGS.device if hasattr(FLAGS, 'device') else \
            ('cuda' if torch.cuda.is_available() else 'cpu')

        self.param_names = list(self.ds.keys())

        self._initialize_param_options()

        self._initialize_network()

        self.gamma = 0.99
        self.epsilon = 0.5
        self.epsilon_decay_step = 1500
        self.huber_loss = nn.HuberLoss()
        self.quality_history = {}
        self.action_probs_history = []
        self.action_probs_history_0 = []
        self.action_probs_history_1 = []
        self.action_history = []
        self.critic_value_history = []
        self.state_history = []
        self.rewards_history = []
        self.episode_rewards = []

        self.solution_mark = {}
        self.solution_set = {}

        self.max_episodes = self.result_number
        self.max_steps_per_episode = self.stop_cond // self.result_number
        self.timeout = getattr(FLAGS, 'ironman_timeout', 60 * 60)

        self.log.info(f"Initialize the IRONMAN-Pro explorer: {kernel_name}")
        self.log.info(f"action dims: {self.action_dims}")

        if run_dse:
            self.run()
            self._save_results()

    def _initialize_param_options(self):
        self.action_dims = []
        self.param_options = {}

        defaults_dict, config_options, _ = self.get_config_dafault_options()
        self.param_options = config_options

        for param_name in self.param_names:
            options = config_options[param_name]
            self.action_dims.append(len(options))

    def _initialize_network(self):
        self.state_dim = (
                len(self.param_names) +
                4 +
                3
        )

        self.ac_network = ActorCriticNetwork(
            state_dim=self.state_dim,
            action_dims=self.action_dims,
            device=self.device
        )

        self.log.info(f"State_dim: {self.state_dim}")

    def _create_state(self, point: Dict[str, Any],
                      last_result: Optional[Result] = None) -> torch.Tensor:
        state = []

        for param_name in self.param_names:
            value = point.get(param_name, 0)
            options = self.param_options[param_name]

            if value in options:
                normalized = options.index(value) / max(len(options) - 1, 1)
            else:
                normalized = 0.0
            state.append(normalized)

        if last_result and hasattr(last_result, 'res_util'):
            for key in ['util-BRAM', 'util-DSP', 'util-LUT', 'util-FF']:
                if key in last_result.res_util:
                    value = last_result.res_util[key]
                    if isinstance(value, (int, float)):
                        state.append(min(value, 1.0))
                    else:
                        state.append(0.5)
                else:
                    state.append(0.5)
        else:
            state.extend([0.5] * 4)

        best_perf_norm = 1.0 / (1.0 + self.best_result.quality / 100.0)
        best_perf_norm = 0.0
        state.append(best_perf_norm)

        progress = min(self.explored_point / self.stop_cond, 1.0)
        state.append(progress)

        design_ratio = len(self.key_perf_dict) / max(self.num_top_designs, 1)
        state.append(min(design_ratio, 1.0))

        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def _select_action(self, state: torch.Tensor, epsilon: float) -> Tuple[List[int], List[float]]:
        actions = []
        probs_for_is = []

        with torch.no_grad():
            action_probs, _ = self.ac_network(state.unsqueeze(0))

        for i, probs in enumerate(action_probs):
            probs = probs.squeeze()

            if np.random.rand() < epsilon:
                action = np.random.randint(0, len(probs))
                probs_for_is.append(1.0 / len(probs))
            else:
                probs_np = probs.cpu().numpy()
                action = np.random.choice(len(probs), p=probs_np)
                probs_for_is.append(probs_np[action])

            actions.append(action)

        return actions, probs_for_is

    def _compute_reward(self, result: Result) -> float:
        if not hasattr(result, 'perf') or result.perf is None:
            return -1.0

        reward = 0.0
        perf_quality = result.quality
        area_quality = result.area
        reward += (100 - perf_quality) / 100
        reward -= (100 - area_quality) / 100

        return reward

    def _update_network(self):
        if len(self.state_history) == 0:
            return
        discounted_returns = []
        G = 0.0

        for reward in reversed(self.rewards_history):
            G = float(reward) + self.gamma * G
            discounted_returns.insert(0, G)

        returns = torch.tensor(discounted_returns, dtype=torch.float32, device=self.device)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        states = torch.stack(self.state_history).to(dtype=torch.float32)

        with torch.no_grad():
            _, values = self.ac_network(states)
            values = values.squeeze().to(dtype=torch.float32)


        advantages = (returns - values).to(dtype=torch.float32)

        actor_losses = []
        critic_losses = []

        batch_start = 0
        for i, actions in enumerate(self.action_history):
            state = states[i:i + 1]

            action_probs, value = self.ac_network(state)
            value = value.squeeze()

            for j, action_idx in enumerate(actions):
                probs = action_probs[j].squeeze()

                log_prob = torch.log(probs[action_idx] + 1e-10)

                prob_0 = float(self.action_probs_history_0[batch_start + j])
                prob_1 = float(self.action_probs_history_1[batch_start + j])
                is_weight = min(prob_1 / (prob_0 + 1e-10), 1.0)
                is_weight = torch.tensor(is_weight, dtype=torch.float32, device=self.device)


                actor_loss = -log_prob * advantages[i].detach() * is_weight
                actor_losses.append(actor_loss)

            batch_start += len(actions)

            critic_loss = self.huber_loss(value, returns[i].detach())
            critic_losses.append(critic_loss)

        total_loss = torch.stack([l.to(dtype=torch.float32) for l in actor_losses]).sum() + \
                     torch.stack([l.to(dtype=torch.float32) for l in critic_losses]).sum()

        self.ac_network.optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), 1.0)

        self.ac_network.optimizer.step()

    def _actions_to_point(self, actions: List[int]) -> Dict[str, Any]:
        point = {}
        for i, param_name in enumerate(self.param_names):
            options = self.param_options[param_name]
            idx = min(actions[i], len(options) - 1)
            point[param_name] = options[idx]
        return point

    def _track_quality_history(self):
        current_qualities = []
        for (quality, point_key) in self.best_results_dict.keys():
            current_qualities.append(quality)

        current_qualities.sort()

        self.quality_history[self.explored_point] = current_qualities

        return current_qualities

    def run(self):
        self.log.info(f"Starting to explore: {self.kernel_name}")

        timer = time.time()
        episode = 0
        epsilon_reduced = False

        try:
            while episode < self.max_episodes and self.explored_point <= self.stop_cond:
                if (time.time() - timer) >= self.timeout:
                    self.log.info(f"Reach the timeout limit {self.timeout}")
                    break

                state = self._reset_episode()
                episode_reward = 0

                if episode > self.epsilon_decay_step and not epsilon_reduced:
                    self.epsilon *= 0.998
                    epsilon_reduced = True
                    self.log.info(f"Epsilon attenuated to: {self.epsilon}")

                for step in range(self.max_steps_per_episode):
                    actions, probs_0 = self._select_action(state, self.epsilon)

                    self.action_probs_history_0.extend(probs_0)

                    point = self._actions_to_point(actions)
                    results = self.get_results([point])

                    if results and len(results) > 0:
                        result = results[0]
                        self.update_best(result)
                        reward = self._compute_reward(result)
                        self.explored_point += 1
                        self._track_quality_history()
                    else:
                        reward = -1.0
                        result = None

                    self.state_history.append(state)
                    self.action_history.append(actions)
                    self.rewards_history.append(reward)
                    episode_reward += reward

                    with torch.no_grad():
                        action_probs, _ = self.ac_network(state.unsqueeze(0))
                        probs_1 = [
                            probs.squeeze()[actions[i]].item()
                            for i, probs in enumerate(action_probs)
                        ]
                    self.action_probs_history_1.extend(probs_1)

                    state = self._create_state(point, result)

                    if step >= self.max_steps_per_episode - 1:
                        break

                self._update_network()

                self.action_probs_history.clear()
                self.action_probs_history_0.clear()
                self.action_probs_history_1.clear()
                self.action_history.clear()
                self.critic_value_history.clear()
                self.state_history.clear()
                self.rewards_history.clear()

                self.episode_rewards.append(episode_reward)


                if episode % 100 == 0:
                    self.log.info(f"Episode {episode}/{self.max_episodes}")
                    self.log.info(f"  Explored point: {self.explored_point}")
                    if self.best_result:
                        self.log.info(f"  Best perf: {self.best_result.perf}")
                    avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                    self.log.info(f"  Avg_reward: {avg_reward:.3f}")

                if episode % 2000 == 0:
                    self.ac_network.lr_scheduler.step()

                episode += 1

            self.log.info(f"\nFinish Exploration！")
            self.log.info(f"Total exploration point: {self.explored_point}")
            self.log.info(f"Find best results: {len(self.best_results_dict)} 个")

        except Exception as e:
            self.log.error(f"Fail to explore: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _reset_episode(self) -> torch.Tensor:
        init_point = self._generate_random_point()
        results = self.get_results(init_point)

        if results and len(results) > 0:
            self.update_best(results[0])
            return self._create_state(init_point[0], results[0])
        else:
            return self._create_state(init_point[0], None)

    def _generate_random_point(self):
        defaults_dict, config_options, config_cond = self.get_config_dafault_options()
        # init population
        init_cand = []
        init_solution = {}
        for key, value in config_options.items():
            init_solution[key] = value[random.randint(0, len(value) - 1)]
        init_cand.append(init_solution)
        return init_cand

    def get_config_dafault_options(self):
        defaults_dict = {key: self.ds[key].default for key in self.param_names}
        config_options = {}
        config_cond = {}

        for key in self.param_names:
            if self.ds[key].deps:
                value = self.ds[key].option_expr.split('if')
                config_options[key] = eval(value[0] + ']')
                config_cond[key] = value[1][:-1]
            else:
                config_options[key] = eval(self.ds[key].option_expr)
                config_cond[key] = ''

        for i, j in config_options.items():
            if '' in j:
                j.remove('')

        return defaults_dict, config_options, config_cond

    def _save_results(self):

        self.log.info('Best Results Found:')
        i = 1

        with open(join('/home/test/xulei/sober/Best_result/AC/test',
                       f'{self.kernel_name}.pickle'), 'wb') as handle:
            pickle.dump(self.best_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
        for _, result in sorted(self.best_results_dict.items()):
            attrs = vars(result)
            self.log.info(f'Design {i}')
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
            i += 1

from torch.optim.lr_scheduler import ExponentialLR


class PolicyGradientNetwork(nn.Module):

    def __init__(self, num_inputs: int, num_actions_list: List[int], device: str = 'cuda:0'):
        super().__init__()
        self.device = device
        self.num_actions_list = num_actions_list

        num_hidden_1 = 128
        num_hidden_2 = 64

        self.fc1 = nn.Linear(num_inputs, num_hidden_1)
        self.fc2 = nn.Linear(num_hidden_1, num_hidden_1)
        self.fc3 = nn.Linear(num_hidden_1, num_hidden_2)

        self.action_heads = nn.ModuleList([
            nn.Linear(num_hidden_2, num_actions)
            for num_actions in num_actions_list
        ])

        self.to(device)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.3)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.1)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.005)

        action_probs = []
        for head in self.action_heads:
            probs = F.softmax(head(x), dim=-1)
            action_probs.append(probs)

        return action_probs


class PGExplorer(Explorer):

    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str,
                 run_dse: bool = True, prune_invalid: bool = False,
                 point: Optional[Dict[str, Any]] = None):

        super().__init__(path_kernel, kernel_name, path_graph, False, prune_invalid)

        if 'evaluate' not in self.config:
            self.config['evaluate'] = {}

        self.config['evaluate']['max-util'] = {
            'BRAM': 0.8,
            'DSP': 0.8,
            'LUT': 0.8,
            'FF': 0.8
        }

        self.device = FLAGS.device if hasattr(FLAGS, 'device') else \
            ('cuda' if torch.cuda.is_available() else 'cpu')

        self.param_names = list(self.ds.keys())

        self.gamma = 0.99
        self.alpha = 0.002
        self.lambda0 = 0.996
        self.lr = 3e-4
        self.epsilon = 0.5
        self.epsilon_threshold = 1500
        self.quality_history = {}
        self.solution_mark = dict()
        self.solution_set = dict()
        self.max_solutions_per_config = 5

        self.num_per_dsp = 2
        self._load_or_generate_dsp_targets()

        self._initialize_param_options()

        self._initialize_policy_network()

        self.action_probs_history = []
        self.action_probs_history_0 = []
        self.action_probs_history_1 = []
        self.action_history = []
        self.state_history = []
        self.rewards_history = []
        self.episode_rewards = []
        self.running_reward = 0

        self.max_episodes = self.stop_cond
        self.max_steps_per_episode = self.result_number
        self.timeout = 60 * 60

        self.p00 = [0.1, 0.9]

        self.log.info(f"Initializing Policy Gradient SAC Explorer: {kernel_name}")
        self.log.info(f"Design space size: {self.ds_size}")
        self.log.info(f"Device: {self.device}")

        if run_dse:
            self.run()
            self._save_results()

    def _load_or_generate_dsp_targets(self):
        try:
            with open('graph_index_dsp_target', 'rb') as fp:
                self.g_ind_dsp_tar = pickle.load(fp)
            self.total_trials = len(self.g_ind_dsp_tar)
            self.log.info(f"Loaded {self.total_trials} DSP target configurations")
        except:
            self.g_ind_dsp_tar = []
            for i in range(10):
                for dsp in [20, 40, 60, 80, 100]:
                    self.g_ind_dsp_tar.append([i, dsp])
            self.total_trials = len(self.g_ind_dsp_tar)
            self.log.info(f"Generated {self.total_trials} DSP target configurations")

    def _initialize_param_options(self):
        self.action_dims = []
        self.param_options = {}

        defaults_dict, config_options, _ = self.get_config_dafault_options()
        self.param_options = config_options

        for param_name in self.param_names:
            options = config_options.get(param_name, [1, 2, 4, 8, 16, 32])
            self.action_dims.append(len(options))

    def _initialize_policy_network(self):
        self.num_inputs = 210

        self.policy_net = PolicyGradientNetwork(
            num_inputs=self.num_inputs,
            num_actions_list=self.action_dims,
            device=self.device
        )

        self.optimizer = Adam(self.policy_net.parameters(), lr=self.lr)
        self.lr_scheduler = ExponentialLR(
            optimizer=self.optimizer,
            gamma=0.9
        )

        self.log.info(f"Policy network initialized - Input dimension: {self.num_inputs}")

    def _create_state(self, point: Dict[str, Any],
                      result: Optional[Result] = None,
                      dsp_target: int = 50,
                      non_assigned_counter: int = 0) -> torch.Tensor:
        state = []

        for param_name in self.param_names:
            value = point.get(param_name, 0)
            options = self.param_options[param_name]

            if value in options:
                normalized = options.index(value) / max(len(options) - 1, 1)
            else:
                normalized = 0.0
            state.append(normalized)

        state.append(dsp_target / 100.0)
        state.append(non_assigned_counter / 100.0)

        if result and hasattr(result, 'perf'):
            perf_norm = 1.0 / (1.0 + result.perf / 10000.0)
        else:
            perf_norm = 0.5
        state.append(perf_norm)

        if result and hasattr(result, 'res_util'):
            for key in ['util-BRAM', 'util-DSP', 'util-LUT', 'util-FF']:
                util_value = result.res_util.get(key, 0.5)
                if isinstance(util_value, (int, float)):
                    util_value = np.clip(util_value, 0, 1)
                else:
                    util_value = 0.5
                state.append(util_value)
        else:
            state.extend([0.5] * 4)

        while len(state) < self.num_inputs:
            if self.episode_rewards:
                state.append(np.tanh(np.mean(self.episode_rewards[-10:]) / 10.0))
            else:
                state.append(0.0)

        state = state[:self.num_inputs]

        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def _select_action(self, state: torch.Tensor,
                       non_assigned_counter: int,
                       target_dsp: int) -> Tuple[List[int], List[float], List[float]]:
        actions = []
        probs_0 = []
        probs_1 = []

        with torch.no_grad():
            action_probs = self.policy_net(state.unsqueeze(0))

        for i, probs in enumerate(action_probs):
            probs = probs.squeeze()
            probs_np = probs.cpu().numpy()

            if np.random.rand() > (1 - self.epsilon):
                action = np.random.randint(0, len(probs))
                probs_0.append(np.clip(probs_np[action], 1e-15, 1.0))
            elif non_assigned_counter >= 2 * target_dsp + 20:
                if len(probs_np) == 2:
                    action = np.random.choice(2, p=self.p00)
                    probs_0.append(self.p00[action])
                else:
                    action = np.random.choice(len(probs_np), p=probs_np)
                    probs_0.append(probs_np[action])
            else:
                action = np.random.choice(len(probs_np), p=probs_np)
                probs_0.append(np.clip(probs_np[action], 1e-15, 1.0))

            probs_1.append(np.clip(probs_np[action], 1e-15, 1.0))
            actions.append(action)

        return actions, probs_0, probs_1

    def _compute_reward(self, result: Result, lut: float, dsp: float, cp: float) -> float:
        reward = 0.0

        reward += (100 - lut) / 100.0 * 10

        reward -= (100 - dsp) / 100.0 * 5

        reward += (100 - cp) / 100.0 * 10

        return reward

    def _update_solution_set(self, graph_index: int, dsp: int, lut: int,
                             action_history: List[List[int]]):
        key = (graph_index, dsp)

        if key in self.solution_mark:
            if len(self.solution_mark[key]) == self.max_solutions_per_config:
                if lut < max(self.solution_mark[key]):
                    max_index = self.solution_mark[key].index(max(self.solution_mark[key]))
                    self.solution_mark[key].pop(max_index)
                    self.solution_set[key].pop(max_index)
                    self.solution_mark[key].append(lut)
                    self.solution_set[key].append(action_history.copy())
            else:
                self.solution_mark[key].append(lut)
                self.solution_set[key].append(action_history.copy())
        else:
            self.solution_mark[key] = [lut]
            self.solution_set[key] = [action_history.copy()]

    def _update_policy_network(self):
        if not self.rewards_history:
            return

        returns = []
        discounted_sum = 0
        for r in reversed(self.rewards_history):
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_losses = []

        for i in range(len(self.state_history)):
            state = self.state_history[i]
            action_probs = self.policy_net(state.unsqueeze(0))

            log_prob = 0
            for j, action_idx in enumerate(self.action_history[i]):
                probs = action_probs[j].squeeze()
                if not torch.isnan(probs).any():
                    log_prob += torch.log(torch.clamp(probs[action_idx], min=1e-15))
                else:
                    log_prob += torch.log(torch.tensor(0.5, device=self.device))

            policy_losses.append(-log_prob * returns[i])

        loss = torch.stack(policy_losses).mean()

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

    def _reset_episode(self) -> torch.Tensor:
        init_point = self._generate_random_point()
        results = self.get_results(init_point)

        if results and len(results) > 0:
            self.update_best(results[0])
            return self._create_state(init_point[0], results[0])
        else:
            return self._create_state(init_point[0], None)

    def _track_quality_history(self):
        current_qualities = []
        for (quality, point_key) in self.best_results_dict.keys():
            current_qualities.append(quality)

        current_qualities.sort()

        self.quality_history[self.explored_point] = current_qualities

        return current_qualities

    def run(self):
        self.log.info(f"Starting Policy Gradient exploration: {self.kernel_name}")

        timer = time.time()
        episode = 0
        ff = 0

        try:
            while self.explored_point < self.stop_cond and episode < self.max_episodes:
                if (time.time() - timer) >= self.timeout:
                    self.log.info(f"Timeout reached {self.timeout} seconds")
                    break

                ind = int(np.remainder(np.floor(episode / self.num_per_dsp), self.total_trials))
                graph_index = int(self.g_ind_dsp_tar[ind][0])
                dsp_target = int(self.g_ind_dsp_tar[ind][1])

                state = self._reset_episode()

                episode_reward = 0
                non_assigned_counter = 0

                if episode > self.epsilon_threshold and ff == 0:
                     self.epsilon = self.epsilon * 0.998
                     ff = 1
                     self.log.info(f"Epsilon decayed to: {self.epsilon}")
                for timestep in range(1, self.max_steps_per_episode):
                    actions, probs_0, probs_1 = self._select_action(
                            state, non_assigned_counter, dsp_target
                        )
                    self.state_history.append(state)
                    self.action_probs_history_0.extend(probs_0)
                    self.action_probs_history_1.extend(probs_1)
                    self.action_history.append(actions)

                    new_point = self._action_indices_to_point(actions)

                    results = self.get_results([new_point])

                    if results and len(results) > 0:
                        result = results[0]
                        self.update_best(result)

                        lut = result.res_util.get('util-LUT', 0.5) * 100 if hasattr(result, 'res_util') else 50
                        dsp = result.res_util.get('util-DSP', 0.5) * 100 if hasattr(result, 'res_util') else 50
                        cp = result.perf if hasattr(result, 'perf') else 10000

                        reward = self._compute_reward(result, lut, dsp, cp)
                        self.rewards_history.append(reward)
                        episode_reward += reward

                        self._update_solution_set(graph_index, int(dsp), int(lut),
                                                      self.action_history)

                        state = self._create_state(new_point, result, dsp_target,
                                                       non_assigned_counter)

                        non_assigned_counter += 1
                        self.explored_point += 1
                        self._track_quality_history()

                        self.log.debug(f"LUT: {lut:.1f}, DSP: {dsp:.1f}, CP: {cp}")

                        if timestep >= self.max_steps_per_episode - 1:
                            break
                    else:
                        reward = -1.0
                        self.rewards_history.append(reward)
                        episode_reward += reward
                        break

                    self._update_policy_network()

                self.running_reward = 0.05 * episode_reward + 0.95 * self.running_reward

                self.action_probs_history.clear()
                self.action_probs_history_0.clear()
                self.action_probs_history_1.clear()
                self.action_history.clear()
                self.rewards_history.clear()
                self.state_history.clear()

                self.episode_rewards.append(episode_reward)
                episode += 1

                if episode % 10 == 0:
                    self.log.info(f"Running reward: {self.running_reward:.2f} at episode {episode}")
                    self.log.info(f"  Solutions found: {sum(len(v) for v in self.solution_mark.values())}")

            self.log.info(f"\nExploration completed!")
            self.log.info(f"Total Episodes: {episode}")
            self.log.info(f"Total explored points: {self.explored_point}")
            self.log.info(f"Solution configurations found: {len(self.solution_mark)}")

        except Exception as e:
            self.log.error(f"Exploration failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _generate_random_point(self):
        defaults_dict, config_options, config_cond = self.get_config_dafault_options()
        init_cand = []
        init_solution = {}
        for key, value in config_options.items():
            init_solution[key] = value[random.randint(0, len(value) - 1)]
        init_cand.append(init_solution)
        return init_cand

    def _action_indices_to_point(self, action_indices: List[int]) -> Dict[str, Any]:
        point = {}
        for i, param_name in enumerate(self.param_names):
            options = self.param_options[param_name]
            idx = min(action_indices[i], len(options) - 1)
            point[param_name] = options[idx]
        return point

    def get_config_dafault_options(self):
        defaults_dict = {key: self.ds[key].default for key in self.param_names}
        config_options = {}
        config_cond = {}

        for key in self.param_names:
            if self.ds[key].deps:
                value = self.ds[key].option_expr.split('if')
                config_options[key] = eval(value[0] + ']')
                config_cond[key] = value[1][:-1]
            else:
                config_options[key] = eval(self.ds[key].option_expr)
                config_cond[key] = ''

        for i, j in config_options.items():
            if '' in j:
                j.remove('')

        return defaults_dict, config_options, config_cond

    def _save_results(self):
        import os

        self.log.info('Best Results Found:')
        i = 1

        with open(join('/home/test/xulei/sober/Best_result/PG/test',
                       f'{self.kernel_name}.pickle'), 'wb') as handle:
            pickle.dump(self.best_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
        for _, result in sorted(self.best_results_dict.items()):
            attrs = vars(result)
            self.log.info(f'Design {i}')
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
            i += 1

        self.log.info(f'  Total explored points: {self.explored_point}')

from copy import deepcopy
from collections import Counter
import math
class LatticeExplorer(Explorer):

    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str,
                 run_dse: bool = True, prune_invalid: bool = False,
                 point: Optional[Dict[str, Any]] = None):

        super().__init__(path_kernel, kernel_name, path_graph, False, prune_invalid)

        if 'evaluate' not in self.config:
            self.config['evaluate'] = {}

        self.config['evaluate']['max-util'] = {
            'BRAM': 0.8,
            'DSP': 0.8,
            'LUT': 0.8,
            'FF': 0.8
        }

        self.initial_sampling_size = getattr(FLAGS, 'lattice_initial_samples', 30)
        self.sphere_radius = getattr(FLAGS, 'lattice_sphere_radius', 0.5)
        self.max_iterations = getattr(FLAGS, 'lattice_max_iterations', 50)
        self.timeout = getattr(FLAGS, 'lattice_timeout', 3600)
        self.quality_history = {}
        self.adaptive_beta = True
        self.beta_alpha = 0.5
        self.beta_beta = 0.5
        self.beta_update_rate = 0.05

        self.exploration_strategies = [
            'pareto_based',
            'random_jump',
            'density_based',
        ]
        self.strategy_weights = [0.5, 0.25, 0.25]

        self._initialize_design_space_params()

        self.explored_configs = []
        self.pareto_frontier = []
        self.adrs_evolution = []
        self.max_distance = self._calculate_max_distance()

        self.config_performance = {}
        self.exploration_density = {}
        self.performance_gradient = {}

        self.iteration_history = []
        self.radius_history = []
        self.strategy_usage = Counter()

        self.reference_result = None

        self.log.info(f"Initializing optimized Lattice Explorer: {kernel_name}")
        self.log.info(f"Design space size: {self.ds_size}")
        self.log.info(f"Resource constraints: All resource types capped at 80%")

        if run_dse:
            self.run()
            self._save_results()
        else:
            if point is not None:
                results = self.get_results([point])
                if results and len(results) > 0:
                    attrs = vars(results[0])
                    self.log.info(', '.join("%s: %s" % item for item in attrs.items()))

    def _initialize_design_space_params(self):
        self.param_names = list(self.ds.keys())
        self.param_options = {}
        self.discretized_descriptor = []

        for param_name in self.param_names:
            options = self.gen_options({p: self.ds[p].default for p in self.param_names}, param_name)

            if not options:
                options = [1, 2, 4, 8, 16]

            types = set(type(opt) for opt in options)
            if len(types) > 1:
                try:
                    options = [float(opt) for opt in options]
                except (ValueError, TypeError):
                    print(f"Warning: Parameter {param_name} has mixed types in options")

            self.param_options[param_name] = options

            if len(options) > 1:
                if all(isinstance(opt, (int, float)) for opt in options):
                    step_size = (options[-1] - options[0]) / (len(options) - 1)
                else:
                    step_size = 1.0
            else:
                step_size = 1.0

            self.discretized_descriptor.append((options, step_size))

    def _calculate_max_distance(self) -> float:
        max_dist = 0
        for options, _ in self.discretized_descriptor:
            if len(options) > 1:
                if all(isinstance(opt, (int, float)) for opt in options):
                    max_dist += (options[-1] - options[0]) ** 2
                else:
                    max_dist += (len(options) - 1) ** 2
        return math.sqrt(max_dist)

    def _calculate_distance(self, config1: List[float], config2: List[float]) -> float:
        dist = 0
        for i in range(len(config1)):
            options, _ = self.discretized_descriptor[i]
            if len(options) > 1:
                val1, val2 = config1[i], config2[i]

                if type(val1) != type(val2):
                    try:
                        val1 = float(val1)
                        val2 = float(val2)
                    except (ValueError, TypeError):
                        try:
                            idx1 = options.index(val1) if val1 in options else 0
                            idx2 = options.index(val2) if val2 in options else 0
                            normalized_dist = (idx1 - idx2) / max(len(options) - 1, 1)
                            dist += normalized_dist ** 2
                            continue
                        except:
                            normalized_dist = 0
                            dist += normalized_dist ** 2
                            continue

                if all(isinstance(opt, (int, float)) for opt in options):
                    range_val = options[-1] - options[0]
                    if range_val > 0:
                        normalized_dist = (val1 - val2) / range_val
                    else:
                        normalized_dist = 0
                else:
                    try:
                        idx1 = options.index(val1) if val1 in options else 0
                        idx2 = options.index(val2) if val2 in options else 0
                        normalized_dist = (idx1 - idx2) / max(len(options) - 1, 1)
                    except:
                        normalized_dist = 0

                dist += normalized_dist ** 2
        return math.sqrt(dist)

    def _config_to_point(self, config: List[float]) -> Dict[str, Any]:
        point = {}
        for i, param_name in enumerate(self.param_names):
            point[param_name] = config[i]
        return point

    def _point_to_config(self, point: Dict[str, Any]) -> List[float]:
        config = []
        for param_name in self.param_names:
            config.append(point[param_name])
        return config

    def _adaptive_beta_sampling(self, n_samples: int) -> List[List[float]]:
        samples = []

        for _ in range(n_samples):
            config = []
            for i, (options, _) in enumerate(self.discretized_descriptor):
                beta_value = np.random.beta(self.beta_alpha, self.beta_beta)

                if i in self.performance_gradient:
                    gradient = self.performance_gradient[i]
                    beta_value = np.clip(beta_value + gradient * 0.1, 0, 1)

                idx = int(beta_value * (len(options) - 1))
                idx = min(idx, len(options) - 1)
                config.append(options[idx])

            samples.append(config)

        return samples

    def _select_exploration_strategy(self) -> str:
        strategy = np.random.choice(
            self.exploration_strategies,
            p=self.strategy_weights
        )
        self.strategy_usage[strategy] += 1
        return strategy

    def _explore_with_strategy(self, strategy: str) -> Optional[List[float]]:
        if strategy == 'pareto_based':
            return self._pareto_based_exploration()
        elif strategy == 'random_jump':
            return self._random_jump_exploration()
        elif strategy == 'density_based':
            return self._density_based_exploration()
        elif strategy == 'gradient_based':
            return self._gradient_based_exploration()
        else:
            return None

    def _pareto_based_exploration(self) -> Optional[List[float]]:
        if not self.pareto_frontier:
            return None

        config = random.choice(self.pareto_frontier)

        return self._perturb_config(list(config.values()))

    def _random_jump_exploration(self) -> Optional[List[float]]:
        config = []
        for options, _ in self.discretized_descriptor:
            if np.random.random() < 0.3:
                choice = random.choice([options[0], options[-1]])
            else:
                choice = random.choice(options)

            if hasattr(choice, 'item'):
                choice = choice.item()
            elif not isinstance(choice, (int, float)):
                choice = float(choice) if not isinstance(choice, str) else choice

            config.append(choice)

        if tuple(config) not in self.explored_configs:
            return config
        return self._perturb_config(config)

    def _density_based_exploration(self) -> Optional[List[float]]:
        if len(self.explored_configs) < 10:
            return self._random_jump_exploration()

        candidates = []
        for _ in range(20):
            config = []
            for options, _ in self.discretized_descriptor:
                config.append(random.choice(options))
            candidates.append(config)

        min_distances = []
        for candidate in candidates:
            min_dist = float('inf')
            for explored in self.explored_configs:
                dist = self._calculate_distance(candidate, list(explored))
                min_dist = min(min_dist, dist)
            min_distances.append(min_dist)

        best_idx = np.argmax(min_distances)
        return candidates[best_idx]

    def _gradient_based_exploration(self) -> Optional[List[float]]:
        config = random.choice(self.pareto_frontier)

        improved_config = []
        for val in config:
            if hasattr(val, 'item'):
                improved_config.append(val.item())
            elif isinstance(val, (int, float)):
                improved_config.append(val)
            else:
                improved_config.append(val)
        for dim_idx in range(len(config)):
            options, _ = self.discretized_descriptor[dim_idx]
            current_val = config[dim_idx]
            current_idx = options.index(current_val) if current_val in options else 0

            improvements = []
            for direction in [-1, 1]:
                new_idx = current_idx + direction
                if 0 <= new_idx < len(options):
                    test_config = config.copy()
                    test_config[dim_idx] = options[new_idx]

                    config_tuple = tuple(test_config)
                    if config_tuple in self.config_performance:
                        perf = self.config_performance[config_tuple]
                        if perf != float('inf'):
                            improvements.append((options[new_idx], perf))

            if improvements:
                best_val = min(improvements, key=lambda x: x[1])[0]
                improved_config[dim_idx] = best_val

        return improved_config

    def _perturb_config(self, config: List[float], num_changes: int = None) -> List[float]:
        if num_changes is None:
            num_changes = random.randint(1, max(1, len(config) // 3))

        new_config = config.copy()
        dimensions = random.sample(range(len(config)), num_changes)

        for dim_idx in dimensions:
            options, _ = self.discretized_descriptor[dim_idx]
            current_val = config[dim_idx]

            try:
                current_idx = options.index(current_val) if current_val in options else 0
            except:
                current_idx = 0

            std = len(options) * 0.15
            offset = int(np.random.normal(0, std))
            new_idx = int(np.clip(current_idx + offset, 0, len(options) - 1))

            new_val = options[new_idx]
            if hasattr(new_val, 'item'):
                new_val = new_val.item()

            new_config[dim_idx] = new_val

        return new_config

    def _save_results(self):
        import os
        os.makedirs('/home/test/xulei/sober/Best_result/Lattice/test', exist_ok=True)

        self.log.info('Best Results Found:')
        i = 1

        with open(join('/home/test/xulei/sober/Best_result/Lattice/test',
                       f'{self.kernel_name}.pickle'), 'wb') as handle:
            pickle.dump(self.best_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()

        for _, result in sorted(self.best_results_dict.items()):
            attrs = vars(result)
            self.log.info(f'Design {i}')
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
            i += 1

        self.log.info(f'\nLattice exploration statistics:')
        self.log.info(f'  Total explored points: {self.explored_point}')
        self.log.info(f'  Pareto frontier size: {len(self.pareto_frontier)}')

    def get_config_dafault_options(self):
        defaults_dict = {key: self.ds[key].default for key in self.ordered_pids}
        config_options = {}
        config_cond = {}
        for key in self.ordered_pids:
            if self.ds[key].deps:
                value = self.ds[key].option_expr.split('if')
                config_options[key] = eval(value[0] + ']')
                config_cond[key] = value[1][:-1]
            else:
                config_options[key] = eval(self.ds[key].option_expr)
                config_cond[key] = ''
        for i, j in config_options.items():
            if '' in j:
                j.remove('')
        return defaults_dict, config_options, config_cond

    def trans_config(self, samples):
        defaults_dict, _, _ = self.get_config_dafault_options()
        key_inx = list(defaults_dict.keys())
        cand = []
        for i in samples:
            temp = deepcopy(defaults_dict)
            for inx, val in enumerate(i):
                if hasattr(val, 'item'):
                    val = val.item()
                temp[key_inx[inx]] = val
            cand.append(temp)
        return cand

    def _track_quality_history(self):
        current_qualities = []
        for (quality, point_key) in self.best_results_dict.keys():
            current_qualities.append(quality)

        current_qualities.sort()

        self.quality_history[self.explored_point] = current_qualities

        return current_qualities

    def run(self) -> None:
        self.log.info(f"Starting optimized Lattice design space exploration: {self.kernel_name}")

        timer = time.time()

        try:
            self.log.info(f"Initial sampling phase")

            samples = []

            beta_samples = self._adaptive_beta_sampling(self.initial_sampling_size // 2)
            samples.extend(beta_samples)

            for _ in range(self.initial_sampling_size // 4):
                config = []
                for options, _ in self.discretized_descriptor:
                    config.append(random.choice(options))
                samples.append(config)

            for _ in range(self.initial_sampling_size // 4):
                config = []
                for options, _ in self.discretized_descriptor:
                    if random.random() < 0.5:
                        config.append(options[0])
                    else:
                        config.append(options[-1])
                samples.append(config)

            cand = self.trans_config(samples)
            r = self.get_results(cand)
            for i in r:
                self.update_best(i)
                self.explored_point += 1
                self._track_quality_history()

            self.pareto_frontier = [i.point for i in self.best_results_dict.values()]
            iteration = 0

            while iteration < self.max_iterations and self.explored_point <= self.stop_cond:
                if (time.time() - timer) >= self.timeout:
                    self.log.info(f"Timeout reached {self.timeout} seconds, stopping exploration")
                    break

                self.log.info(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

                strategy = self._select_exploration_strategy()
                print(f"Using strategy: {strategy}")

                nums = 0
                new_config = []
                while nums <= self.result_number and nums <= self.ds_size:
                    new_config.append(self._explore_with_strategy(strategy))
                    nums += 1
                if len(new_config):
                    new_cand = self.trans_config(new_config)
                    r = self.get_results(new_cand)
                    for i in r:
                        self.update_best(i)
                        self.explored_point += 1
                        self._track_quality_history()

                    self.pareto_frontier = [i.point for i in self.best_results_dict.values()]

                iteration += 1

            self.log.info(f"\nExploration completed!")
            self.log.info(f"Total explored design points: {self.explored_point}")
            self.log.info(f"Final Pareto frontier contains {len(self.pareto_frontier)} points")

        except Exception as e:
            self.log.error(f"Optimized Lattice exploration failed: {e}")
            raise


from random import randint
from sklearn.svm import SVC


class HGBOExplorer(Explorer):
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool = True,
                 prune_invalid=FLAGS.prune_class, point: DesignPoint = None):
        super(HGBOExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.batch_size = 1
        self.log.info('Initialization completed')

        start_time = time.time()
        if self.run_dse:
            self.run()
            attrs = vars(self.best_result)
            self.log.info('Best Results Found:')
            i = 1
            with open(join('/home/test/xulei/sober/Best_result/HGBO-DSE/test', f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
        end_time = time.time()
        print(f'Runtime: {end_time - start_time}')

    def run(self) -> None:
        defaults_dict, config_options, config_cond = self.get_config_dafault_options()

        initialization_phase = self._initialize_apdse_population(config_options)
        mcts_tree = MCTSTree(config_options)

        while self.explored_point <= self.stop_cond:
            self.log.info('-' * 50)
            self.log.info(f'Explored point {self.explored_point}/{self.stop_cond}')
            self.log.info('-' * 50)

            evaluation_results = self.get_results(initialization_phase)
            for result in evaluation_results:
                self.explored_point += 1
                update_flag = self.update_best(result)

            partitioned_population = self._learning_partition_phase(evaluation_results, mcts_tree, config_options)
            selected_population = self._mcts_selection_phase(partitioned_population, mcts_tree, config_options)
            new_population = self._bayesian_sampling_phase(selected_population, evaluation_results, config_options)
            initialization_phase = new_population

        self.log.info(f'Total design points explored: {self.explored_point}')

    def _initialize_apdse_population(self, config_options):
        population_collection = []
        sobol_sequence = self._generate_sobol_sequence(len(config_options), self.result_number)

        for index in range(self.result_number):
            design_configuration = {}
            parameter_keys = list(config_options.keys())

            for j, key in enumerate(parameter_keys):
                sobol_value = sobol_sequence[index][j]
                available_values = config_options[key]
                value_index = min(int(sobol_value * len(available_values)), len(available_values) - 1)
                design_configuration[key] = available_values[value_index]

            population_collection.append(design_configuration)

        return population_collection

    def _learning_partition_phase(self, results, mcts_tree, config_options):
        dominance_scores = self._calculate_dominance_scores(results)
        svm_classifier = self._train_svm_partitioner(results, dominance_scores)
        mcts_tree.update_partitions(svm_classifier, dominance_scores)
        return self._apply_partition_to_population(results, svm_classifier, config_options)

    def _calculate_dominance_scores(self, results):
        score_collection = []
        for i, result_i in enumerate(results):
            dominance_count = 0
            for j, result_j in enumerate(results):
                if i != j and self._dominates(result_j, result_i):
                    dominance_count += 1
            score_collection.append(dominance_count)
        return np.array(score_collection)

    def _dominates(self, result_a, result_b):
        return (result_a.quality >= result_b.quality and
                any(getattr(result_a, attr) > getattr(result_b, attr)
                    for attr in ['perf', 'area'] if hasattr(result_a, attr)))

    def _train_svm_partitioner(self, results, dominance_scores):
        feature_matrix = []
        for result in results:
            design_features = []
            for key in sorted(vars(result).keys()):
                if key not in ['quality', 'perf', 'area']:
                    attribute_value = getattr(result, key)
                    if isinstance(attribute_value, (int, float)):
                        design_features.append(attribute_value)
            feature_matrix.append(design_features)

        feature_matrix = np.array(feature_matrix)
        median_dominance = np.median(dominance_scores)
        performance_labels = dominance_scores <= median_dominance

        svm_model = SVC(kernel='rbf', probability=True)
        svm_model.fit(feature_matrix, performance_labels)

        return svm_model

    def _mcts_selection_phase(self, population, mcts_tree, config_options):
        selected_region = mcts_tree.select_region_uct()
        selected_designs = []

        for design in population:
            if self._is_design_in_region(design, selected_region, config_options):
                selected_designs.append(design)

        while len(selected_designs) < self.result_number:
            random_design = self._generate_random_design(config_options)
            selected_designs.append(random_design)

        return selected_designs[:self.result_number]

    def _bayesian_sampling_phase(self, selected_population, previous_results, config_options):
        gaussian_process_models = self._build_gaussian_process_models(previous_results)
        acquisition_function = self._qehvi_acquisition_function(gaussian_process_models, previous_results)

        new_designs = []
        for i in range(len(selected_population)):
            new_design = self._generate_design_by_acquisition(
                selected_population[i], acquisition_function, config_options)
            new_designs.append(new_design)

        return new_designs

    def _qehvi_acquisition_function(self, gp_models, previous_results):
        def acquisition(new_design):
            return random.random()

        return acquisition

    def _generate_random_population(self, config_options):
        population = []
        for i in range(self.result_number):
            design = {}
            for key, values in config_options.items():
                design[key] = values[randint(0, len(values) - 1)]
            population.append(design)
        return population

    def _generate_random_design(self, config_options):
        design = {}
        for key, values in config_options.items():
            design[key] = values[randint(0, len(values) - 1)]
        return design

    def _is_design_in_region(self, design, region, config_options):
        return True

    def _build_gaussian_process_models(self, results):
        return {}

    def _generate_design_by_acquisition(self, base_design, acquisition_func, config_options):
        new_design = base_design.copy()
        for key in config_options.keys():
            if random.random() < 0.3:
                new_design[key] = config_options[key][randint(0, len(config_options[key]) - 1)]
        return new_design

    def _apply_partition_to_population(self, results, svm_model, config_options):
        return [self._generate_random_design(config_options) for _ in range(self.result_number)]

    def _generate_sobol_sequence(self, dim, n_points):
        sequence = np.random.random((n_points, dim))
        return sequence


class MCTSTree:
    def __init__(self, config_options):
        self.root = MCTSNode(config_options)
        self.nodes = [self.root]

    def update_partitions(self, svm_model, dominance_scores):
        pass

    def select_region_uct(self):
        return self.root.region


class MCTSNode:
    def __init__(self, config_options):
        self.region = config_options
        self.children = []
        self.visit_count = 0
        self.hypervolume = 0
        self.samples = []


class QLMOEAExplorer(Explorer):
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool = True,
                 prune_invalid=FLAGS.prune_class, point: DesignPoint = None):
        super(QLMOEAExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.batch_size = 1
        self.learning_rate = 0.8
        self.discount_factor = 0.5
        self.crossover_rate = 1.0
        self.mutation_rate = 0.8

        self.q_table = {
            'start': {'a0': 0, 'a1': 0, 'a2': 0},
            'improvement': {'a0': 0, 'a1': 0, 'a2': 0},
            'no_improvement': {'a0': 0, 'a1': 0, 'a2': 0}
        }
        self.current_state = 'start'
        self.population_history = []
        self.gene_frequency = {}
        self.log.info('QL-MOEA initialization completed')

        start_time = time.time()
        if self.run_dse:
            self.run()
            with open(join('/home/test/xulei/sober/Best_result/QL-MOEA/test', f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            i = 1
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
        end_time = time.time()
        print(f'QL-MOEA total runtime: {end_time - start_time} seconds')

    def select_mutation_strategy(self) -> str:
        if self.current_state == 'start':
            return random.choice(['a0', 'a1', 'a2'])

        state_actions = self.q_table[self.current_state]
        max_q_value = max(state_actions.values())
        best_actions = [action for action, q_value in state_actions.items() if q_value == max_q_value]
        return random.choice(best_actions)

    def update_q_table(self, reward: float, action: str, next_state: str):
        current_q = self.q_table[self.current_state][action]
        max_next_q = max(self.q_table[next_state].values())
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[self.current_state][action] = new_q
        self.current_state = next_state

    def analyze_gene_frequency(self, population: list):
        if not population:
            return {}

        config_keys = list(population[0].keys())
        frequency_analysis = {key: {} for key in config_keys}

        for individual in population:
            for key, value in individual.items():
                if value not in frequency_analysis[key]:
                    frequency_analysis[key][value] = 0
                frequency_analysis[key][value] += 1

        self.gene_frequency = frequency_analysis
        return frequency_analysis

    def identify_superior_positions(self, count: int) -> list:
        if not self.gene_frequency:
            return random.sample(list(self.config_options.keys()), min(count, len(self.config_options)))

        position_scores = {}
        for key, value_dist in self.gene_frequency.items():
            if value_dist:
                max_freq = max(value_dist.values())
                total_individuals = sum(value_dist.values())
                position_scores[key] = max_freq / total_individuals if total_individuals > 0 else 0

        sorted_positions = sorted(position_scores, key=position_scores.get, reverse=True)
        return sorted_positions[:min(count, len(sorted_positions))]

    def single_point_crossover(self, parent1: dict, parent2: dict) -> dict:
        child = parent1.copy()
        keys = list(parent1.keys())
        if len(keys) <= 1:
            return child

        crossover_point = random.randint(1, len(keys) - 1)
        for i in range(crossover_point, len(keys)):
            child[keys[i]] = parent2[keys[i]]
        return child

    def single_point_mutation(self, individual: dict, strategy: str) -> dict:
        mutated = individual.copy()
        keys = list(individual.keys())

        if strategy == 'a0':
            mutation_point = random.randint(0, len(keys) - 1)
            key = keys[mutation_point]
            available_options = [opt for opt in self.config_options[key] if opt != individual[key]]
            if available_options:
                mutated[key] = random.choice(available_options)

        elif strategy == 'a1':
            superior_pos = self.identify_superior_positions(1)
            protected_key = superior_pos[0] if superior_pos else keys[0]
            for key in keys:
                if key != protected_key and random.random() < self.mutation_rate:
                    available_options = [opt for opt in self.config_options[key] if opt != individual[key]]
                    if available_options:
                        mutated[key] = random.choice(available_options)

        elif strategy == 'a2':
            superior_positions = self.identify_superior_positions(2)
            protected_keys = set(superior_positions[:2]) if len(superior_positions) >= 2 else set(keys[:1])
            for key in keys:
                if key not in protected_keys and random.random() < self.mutation_rate:
                    available_options = [opt for opt in self.config_options[key] if opt != individual[key]]
                    if available_options:
                        mutated[key] = random.choice(available_options)

        return mutated

    def calculate_pareto_front(self, results: list) -> list:
        if not results:
            return []

        pareto_front = []
        for i, result_i in enumerate(results):
            dominated = False
            for j, result_j in enumerate(results):
                if i != j:
                    if self.dominates(result_j, result_i):
                        dominated = True
                        break
            if not dominated:
                pareto_front.append(result_i)

        return pareto_front

    def dominates(self, result_a, result_b) -> bool:
        area_a = getattr(result_a, 'area', float('inf'))
        latency_a = getattr(result_a, 'perf', float('inf'))
        area_b = getattr(result_b, 'area', float('inf'))
        latency_b = getattr(result_b, 'perf', float('inf'))

        return (area_a >= area_b and latency_a >= latency_b) and (area_a > area_b or latency_a > latency_b)

    def calculate_adrs(self, pareto_front: list, reference_front: list) -> float:
        if not pareto_front or not reference_front:
            return float('inf')

        total_distance = 0.0
        for ref_point in reference_front:
            min_distance = float('inf')
            for point in pareto_front:
                quality_diff = abs(getattr(point, 'area', 0) - ref_point)
                distance = (quality_diff) ** 0.5
                min_distance = min(min_distance, distance)
            total_distance += min_distance

        return total_distance / len(reference_front)

    def is_pareto_improved(self, old_front: list, new_front: list, reference_front: list) -> bool:
        if not old_front:
            return True

        old_adrs = self.calculate_adrs(old_front, reference_front)
        new_adrs = self.calculate_adrs(new_front, reference_front)

        return new_adrs < old_adrs

    def get_reference_pareto_front(self) -> list:
        path = join(f'/home/test/xulei/sober/Best_result/ref/{self.kernel_name}.pickle')
        data_list = []
        with open(path, 'rb') as f:
            data = pickle.load(f)
            for i in data:
                data_1, _ = i
                data_list.append(data_1)
        return data_list

    def run(self) -> None:
        defaults_dict, config_options, config_cond = self.get_config_dafault_options()
        self.config_options = config_options

        population = []
        for i in range(self.result_number):
            init_solution = {}
            for key, value in config_options.items():
                init_solution[key] = value[randint(0, len(value) - 1)]
            population.append(init_solution)

        reference_front = self.get_reference_pareto_front()
        previous_pareto = []

        iteration = 0
        while self.explored_point <= self.stop_cond:
            print('=' * 60)
            print(f'QL-MOEA Iteration {iteration}: Explored {self.explored_point}/{self.stop_cond} points')
            print('=' * 60)

            self.analyze_gene_frequency(population)
            results = self.get_results(population)
            current_pareto = self.calculate_pareto_front(results)

            for result in results:
                self.explored_point += 1
                if isinstance(result, Result):
                    fitness_values = [res.quality for res in self.best_results_dict.values()]
                    self.best_save_results[self.explored_point] = fitness_values
                    self.update_best(result)

            mutation_strategy = self.select_mutation_strategy()

            new_population = []
            for i in range(len(population)):
                parent_j = random.choice(population)
                child_crossover = self.single_point_crossover(population[i], parent_j)
                new_population.append(child_crossover)

                child_mutation = self.single_point_mutation(population[i], mutation_strategy)
                new_population.append(child_mutation)

            new_results = self.get_results(new_population[:self.result_number])
            new_pareto = self.calculate_pareto_front(new_results)

            improvement = self.is_pareto_improved(previous_pareto, new_pareto, reference_front)
            reward = 1.0 if improvement else 0.0
            next_state = 'improvement' if improvement else 'no_improvement'

            self.update_q_table(reward, mutation_strategy, next_state)

            print(f"Mutation strategy: {mutation_strategy}, Reward: {reward}, Next state: {next_state}")
            print(f"Q-table state: {dict(self.q_table[self.current_state])}")

            combined_population = population + new_population
            combined_results = results + new_results

            fitness_population = []
            for idx, result in enumerate(combined_results):
                quality = getattr(result, 'quality', float('inf'))
                fitness_population.append((quality, combined_population[idx]))

            sorted_r = sorted(fitness_population, key = lambda x: x[0], reverse=True)
            population = [i for _, i in sorted_r[:5]]
            previous_pareto = new_pareto
            iteration += 1


        print("QL-MOEA exploration completed successfully")

        print(f"Total points explored: {self.explored_point}")


class MOEDAExplorer(Explorer):
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool = True,
                 prune_invalid=FLAGS.prune_class, point: DesignPoint = None):
        super(MOEDAExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)

        self.T = 10
        self.crossover_rate = 1.0
        self.mutation_rate = 0.5
        self.eda_update_rate = 0.2

        self.weight_vectors = []
        self.neighborhoods = []
        self.probability_vectors = []
        self.reference_point = None

        self.log.info('MOEDA initialization completed')
        start_time = time.time()

        if self.run_dse:
            self.run()
            with open(join('/home/test/xulei/sober/Best_result/MOEDA/test', f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            i = 1
            self.log.info('Pareto front designs:')
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))

        end_time = time.time()
        print(f'MOEDA execution time: {end_time - start_time}')

    def initialize_weight_vectors(self):
        for i in range(self.result_number):
            w1 = i / (self.result_number - 1) if self.result_number > 1 else 0.5
            w2 = 1 - w1
            self.weight_vectors.append([w1, w2])

    def initialize_neighborhoods(self):
        for i in range(self.result_number):
            distances = []
            for j in range(self.result_number):
                if i != j:
                    dist = np.linalg.norm(np.array(self.weight_vectors[i]) -
                                          np.array(self.weight_vectors[j]))
                    distances.append((j, dist))

            distances.sort(key=lambda x: x[1])
            neighborhood = [idx for idx, _ in distances[:self.T]]
            self.neighborhoods.append(neighborhood)

    def initialize_probability_vectors(self):
        defaults_dict, config_options, _ = self.get_config_dafault_options()

        for i in range(self.result_number):
            prob_vector = {}
            for key, values in config_options.items():
                num_values = len(values)
                prob_vector[key] = [1.0 / num_values] * num_values
            self.probability_vectors.append(prob_vector)

    def tchebycheff_scalarizing_function(self, solution, weight_vector, reference_point):
        normalized_area = self.normalize_objective(solution.area, 'area')
        normalized_latency = self.normalize_objective(solution.perf, 'perf')

        obj1 = weight_vector[0] * abs(normalized_area - reference_point[0])
        obj2 = weight_vector[1] * abs(normalized_latency - reference_point[1])

        return max(obj1, obj2)

    def normalize_objective(self, value, obj_type):
        if obj_type == 'area':
            min_val = self.min_area if hasattr(self, 'min_area') else value
            max_val = self.max_area if hasattr(self, 'max_area') else value * 2
        else:
            min_val = self.min_latency if hasattr(self, 'min_latency') else value
            max_val = self.max_latency if hasattr(self, 'max_latency') else value * 2

        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)

    def update_probability_vector(self, subproblem_idx):
        neighborhood_solutions = []

        for neighbor_idx in self.neighborhoods[subproblem_idx]:
            if neighbor_idx < len(self.population):
                neighborhood_solutions.append(self.population[neighbor_idx])

        if not neighborhood_solutions:
            return

        _, config_options, _ = self.get_config_dafault_options()
        prob_vector = {}

        for key, values in config_options.items():
            value_counts = {v: 0 for v in values}
            total_count = len(neighborhood_solutions)

            for solution in neighborhood_solutions:
                if key in solution:
                    value = solution[key]
                    if value in value_counts:
                        value_counts[value] += 1

            prob_dist = []
            for value in values:
                prob = value_counts[value] / total_count
                prob_dist.append(prob)

            prob_vector[key] = prob_dist

        self.probability_vectors[subproblem_idx] = prob_vector

    def eda_update_operator(self, solution, subproblem_idx):
        updated_solution = solution.copy()
        prob_vector = self.probability_vectors[subproblem_idx]

        for key in prob_vector.keys():
            if random.random() < self.eda_update_rate:
                values = list(self.get_config_dafault_options()[1][key])
                probabilities = prob_vector[key]

                if sum(probabilities) > 0:
                    chosen_value = random.choices(values, weights=probabilities)[0]
                    updated_solution[key] = chosen_value

        return updated_solution

    def crossover_and_mutation(self, parent1, parent2):
        _, config_options, _= self.get_config_dafault_options()
        child1, child2 = {}, {}

        keys = list(config_options.keys())
        crossover_point = random.randint(1, len(keys) - 1)

        for i, key in enumerate(keys):
            if i < crossover_point:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]

        for child in [child1, child2]:
            for key in keys:
                if random.random() < self.mutation_rate:
                    values = config_options[key]
                    child[key] = random.choice(values)

        return child1, child2

    def update_reference_point(self, solution):
        if self.reference_point is None:
            self.reference_point = [solution.area, solution.perf]
            self.min_area = solution.area
            self.max_area = solution.area
            self.min_latency = solution.perf
            self.max_latency = solution.perf
        else:
            self.reference_point[0] = min(self.reference_point[0], solution.area)
            self.reference_point[1] = min(self.reference_point[1], solution.perf)

            self.min_area = min(self.min_area, solution.area)
            self.max_area = max(self.max_area, solution.area)
            self.min_latency = min(self.min_latency, solution.perf)
            self.max_latency = max(self.max_latency, solution.perf)


    def dominates(self, sol1, sol2):
        return (sol1.area >= sol2.area and sol1.perf >= sol2.perf and
                (sol1.area > sol2.area or sol1.perf > sol2.perf))

    def run(self) -> None:
        self.initialize_weight_vectors()
        self.initialize_neighborhoods()
        self.initialize_probability_vectors()

        self.population = []
        defaults_dict, config_options, config_cond = self.get_config_dafault_options()

        for i in range(self.result_number):
            init_config = {}
            for key, values in config_options.items():
                init_config[key] = random.choice(values)
            self.population.append(init_config)

        initial_results = self.get_results(self.population)
        for i, result in enumerate(initial_results):
            self.explored_point += 1
            self.update_reference_point(result)
            self.update_best(result)

        while self.explored_point <= self.stop_cond:
            self.log.info(f'Exploration point {self.explored_point}/{self.stop_cond}')

            new_population = []
            for i in range(self.result_number):
                neighbors = self.neighborhoods[i]
                if len(neighbors) >= 2:
                    parent_idx1, parent_idx2 = random.sample(neighbors, 2)
                    parent1 = self.population[parent_idx1]
                    parent2 = self.population[parent_idx2]

                    child1, child2 = self.crossover_and_mutation(parent1, parent2)

                    child3 = self.eda_update_operator(self.population[i], i)

                    candidates = [parent1, parent2, child1, child2, child3]
                    candidate_results = self.get_results(candidates)

                    best_solution = None
                    best_score = float('inf')

                    for result in candidate_results:
                        self.explored_point += 1
                        score = self.tchebycheff_scalarizing_function(
                            result, self.weight_vectors[i], self.reference_point)

                        if score < best_score:
                            best_score = score
                            best_solution = {}
                            for key, value in result.point.items():
                                if torch.is_tensor(value):
                                    value = int(value)
                                best_solution[key] = value

                        self.update_reference_point(result)
                        self.update_best(result)

                    new_population.append(best_solution)
                    self.update_probability_vector(i)

            self.population = new_population


        self.log.info(f'MOEDA exploration completed. Total design points explored: {self.explored_point}')


class PSOExplorer(Explorer):
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool = True,
                 prune_invalid=FLAGS.prune_class, point: DesignPoint = None):
        super(PSOExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.batch_size = 1
        self.log.info('Done init')
        start_time = time.time()
        if self.run_dse:
            self.run()
            attrs = vars(self.best_result)
            self.log.info('Best Results Found:')
            i = 1
            with open(join('/home/test/xulei/sober/Best_result/PSO/test', f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
                i += 1
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
        end_time = time.time()
        print(f'runtime: {end_time - start_time}')

    def run(self) -> None:
        defaults_dict, config_options, config_cond = self.get_config_dafault_options()
        w = 0.8
        c1 = 1.5
        c2 = 1.5

        particles = []
        global_best = None
        global_best_fitness = float('-inf')

        for i in range(self.result_number):
            particle = {
                'position': {},
                'velocity': {},
                'best_position': {},
                'best_fitness': float('-inf')
            }
            for key, values in config_options.items():
                rand_idx = random.randint(0, len(values) - 1)
                particle['position'][key] = values[rand_idx]
                particle['velocity'][key] = random.uniform(-1, 1)
            particles.append(particle)

        while self.explored_point <= self.stop_cond:
            print('------------------------------------------------------')
            print(f'explored point {self.explored_point}/{self.stop_cond}')
            print('------------------------------------------------------')

            solutions = [p['position'] for p in particles]
            results = self.get_results(solutions)

            for i, r in enumerate(results):
                self.explored_point += 1
                fitness = [res.quality for res in self.best_results_dict.values()]
                self.best_save_results[self.explored_point] = fitness

                if isinstance(r, Result):
                    attrs = vars(r)
                    self.log.debug(f'Evaluating Design')
                    self.log.debug(', '.join("%s: %s" % item for item in attrs.items()))
                    flag = self.update_best(r)

                current_fitness = r.quality
                if current_fitness > particles[i]['best_fitness']:
                    particles[i]['best_fitness'] = current_fitness
                    particles[i]['best_position'] = copy.deepcopy(particles[i]['position'])

                if current_fitness > global_best_fitness:
                    global_best_fitness = current_fitness
                    global_best = copy.deepcopy(particles[i]['position'])

            for particle in particles:
                for key, values in config_options.items():
                    current_idx = values.index(particle['position'][key])
                    best_idx = values.index(particle['best_position'][key])
                    global_idx = values.index(global_best[key]) if global_best else current_idx

                    r1, r2 = random.random(), random.random()
                    new_velocity = (w * particle['velocity'][key] +
                                    c1 * r1 * (best_idx - current_idx) +
                                    c2 * r2 * (global_idx - current_idx))

                    particle['velocity'][key] = max(-3, min(3, new_velocity))

                    new_idx = round(current_idx + particle['velocity'][key])
                    new_idx = max(0, min(len(values) - 1, new_idx))

                    particle['position'][key] = values[int(new_idx)]

        self.log.info(f'Explored {self.explored_point} points')


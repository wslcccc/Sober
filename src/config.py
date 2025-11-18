from networkx.classes import neighbors
import torch
from src.utils import get_user, get_host
import argparse
from CoGNN.layers import ModelType
TARGETS = ['perf', 'quality', 'util-BRAM', 'util-DSP', 'util-LUT', 'util-FF',
           'total-BRAM', 'total-DSP', 'total-LUT', 'total-FF']
# MACHSUITE_KERNEL = ['gemm-blocked', 'gemm-ncubed']
# # MACHSUITE_KERNEL = []
# poly_KERNEL = ['2mm', '3mm', 'atax', 'bicg', 'bicg-large', 'covariance', 'fdtd-2d', 'fdtd-2d-large', 'gemm-p', 'gemm-p-large', 'gemver',
#                 'gesummv', 'heat-3d', 'jacobi-1d', 'syr2k', 'trmm', 'trmm-opt', 'mvt-medium']



MACHSUITE_KERNEL = ['stencil','nw']


poly_KERNEL = ['jacobi-2d', 'mvt', 'symm',
               'symm-opt', 'syrk', 'correlation',
               'atax-medium', 'bicg-medium']


parser = argparse.ArgumentParser()

parser.add_argument('--model', default='simple')

dataset = 'programl'
parser.add_argument('--dataset', default=dataset)

benchmark = ['machsuite', 'poly']
parser.add_argument('--benchmarks', default=benchmark)

tag = 'whole-machsuite-poly'
parser.add_argument('--tag', default=tag)

# encoder_path = None
encoder_path = '/home/test/xulei/sober/save_models_and_data/encoders.klepto'
parser.add_argument('--encoder_path', default=encoder_path)

# model_path = None
model_path = '/home/test/xulei/sober/save_models_and_data/regression_model_state_dict.pth'
parser.add_argument('--model_path', default=model_path)

parser.add_argument('--num_features', default=153)

# TASK = 'class'
TASK = 'regression'
parser.add_argument('--task', default=TASK)

SUBTASK = 'dse'
# SUBTASK = 'inference'
# SUBTASK = 'train'
parser.add_argument('--subtask', default=SUBTASK)
parser.add_argument('--val_ratio', type=float, default=0.15) # ratio of database for validation set

# explorer = 'Exhastive'
explorer = 'NSGA-II'
# explorer = 'SA'
# explorer = 'ACO'
# explorer = 'PSO'
# explorer = 'lattice'
# explorer =  'AC'
# explorer = 'PG'
# explorer = 'QLMOEA'
#explorer = 'MOEDA'
# explorer = 'APDSE'

parser.add_argument('--explorer', default=explorer)

model_tag = 'test'
parser.add_argument('--model_tag', default=model_tag)

parser.add_argument('--activation', default='elu')

parser.add_argument('--prune_util', default=True)
parser.add_argument('--prune_class', default=False)
# parser.add_argument('--prune_class', default=True)

parser.add_argument('--force_regen', type=bool, default=False)

parser.add_argument('--no_pragma', type=bool, default=False)

pids = ['__PARA__L3', '__PIPE__L2', '__PARA__L1', '__PIPE__L0', '__TILE__L2', '__TILE__L0', '__PARA__L2', '__PIPE__L0']
parser.add_argument('--ordered_pids', default=pids)

# multi_target = ['perf']
multi_target = ['perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']
target = 'perf'
parser.add_argument('--target', default=multi_target)

parser.add_argument('--separate_perf', type = bool, default=False )

parser.add_argument('--num_layers', type=int, default=6)

parser.add_argument('--encode_edge', type=bool, default=False)

parser.add_argument('--loss', type=str, default='MSE')


EPSILON = 1e-3
parser.add_argument('--epsilon', default=EPSILON)
NORMALIZER = 1e7
parser.add_argument('--normalizer', default=NORMALIZER)
# MAX_NUMBER = 3464510.00
MAX_NUMBER = 1e10
parser.add_argument('--max_number', default=MAX_NUMBER)

norm = 'speedup-log2' # 'const' 'log2' 'speedup' 'off' 'speedup-const' 'const-log2' 'none' 'speedup-log2'
parser.add_argument('--norm_method', default=norm)

parser.add_argument('--invalid', type = bool, default=False ) # False: do not include invalid designs

parser.add_argument('--all_kernels', type = bool, default=True)

parser.add_argument('--multi_target', type = bool, default=True)

parser.add_argument('--save_model', type = bool, default=False)

parser.add_argument('--encode_log', type = bool, default=False)

parser.add_argument('--D', type=int, default=64)

batch_size = 64
parser.add_argument('--batch_size', type=int, default=batch_size)

epoch_num = 1500
parser.add_argument('--epoch_num', type=int, default=epoch_num)

device = 'cuda:0'
# device = 'cpu'
parser.add_argument('--device', default=device)

parser.add_argument('--print_every_iter', type=int, default=100)

parser.add_argument('--plot_pred_points', type=bool, default=False)

best_result_path = '/best_result'
parser.add_argument('--best_result_path', type=str, default=best_result_path)

dse_unseen_kernel = ['bicg', 'doitgen', 'gesummv', '2mm']
parser.add_argument('--dse_unseen_kernel', type=list, default=dse_unseen_kernel)

out_dim = 1 if TASK == 'regression' else 2
parser.add_argument('--out_dim', type=int, default=out_dim)

#gumbel
parser.add_argument("--learn_temp", default=False)
parser.add_argument("--temp_model_type", dest="temp_model_type", default=ModelType.LIN,
                        type=ModelType.from_string, choices=list(ModelType))
parser.add_argument("--tau0", default=0.5, type=float)
parser.add_argument("--temp", default=0.01, type=float)

#enviroment
parser.add_argument("--env_model_type", default=ModelType.SUM_GNN,
                        type=ModelType.from_string, choices=list(ModelType))
parser.add_argument("--env_num_layers", default=3, type=int)
parser.add_argument("--env_dim", default=256, type=int)
parser.add_argument("--skip", default=False)
parser.add_argument("--batch_norm", default=True)
parser.add_argument("--layer_norm", default=True)
parser.add_argument("--dec_num_layers", default=1, type=int)
parser.add_argument("--dropout", default=0.1, type=float)

# policy cls parameters
parser.add_argument("--act_model_type", default=ModelType.MEAN_GNN,
                        type=ModelType.from_string, choices=list(ModelType))
parser.add_argument("--act_num_layers", default=2, type=int)
parser.add_argument("--act_dim", default=16, type=int)


# NSGA-II parameter
crossover_mutation_rate = 0.1
parser.add_argument("--crossover_mutation_rate", default=crossover_mutation_rate, type=int)
iter_stop_num = 0.1
parser.add_argument("--iter_stop_num", default=iter_stop_num, type=int)

# SA parameter
initial_temperature = 100
parser.add_argument("--initial_temperature", default=initial_temperature, type=int)
stop_temperature = 0.1
parser.add_argument("--stop_temperature", default=stop_temperature, type=int)
cooling_rate = 0.1
parser.add_argument("--cooling_rate", default=cooling_rate, type=int)
neighbor_distance_rate = 0.3
parser.add_argument("--neighbor_distance_rate", default=neighbor_distance_rate, type=int)


# Sober



"""
Other info.
"""
parser.add_argument('--user', default=get_user())

parser.add_argument('--hostname', default=get_host())

FLAGS = parser.parse_args()

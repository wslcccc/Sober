from torchgen.api.ufunc import kernel_name

from src.programl_data import print_data_stats, _check_any_in_str, NON_OPT_PRAGMAS, WITH_VAR_PRAGMAS, \
    _in_between, _encode_edge_dict, _encode_edge_torch, _encode_X_torch, create_edge_index, ALL_KERNEL
from src.utils import get_root_path, load
from glob import glob
from os.path import join
import networkx as nx
from src.config import FLAGS
import pickle
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
# MACHSUITE_KERNEL = ['gemm-blocked', 'gemm-ncubed']
#
# poly_KERNEL = ['2mm', '3mm', 'atax', 'bicg', 'bicg-large', 'covariance', 'fdtd-2d', 'fdtd-2d-large', 'gemm-p', 'gemm-p-large', 'gemver',
#                'gesummv', 'heat-3d', 'syr2k', 'symm', 'trmm', 'trmm-opt', 'mvt-medium', 'jacobi-1d']

MACHSUITE_KERNEL = ['stencil','nw']

poly_KERNEL = ['jacobi-2d', 'mvt',
               'symm-opt', 'syrk', 'correlation',
               'atax-medium', 'bicg-medium']

alg = ['AC', 'ACO', 'APDSE', 'lattice', 'MOEDA', 'NSGA-II', 'PG', 'PSO', 'QL-MOEA', 'SA']
def encode_node(g, encoder):
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

    enc_ntype = encoder['enc_ntype']
    enc_ptype = encoder['enc_ptype']
    enc_itype = encoder['enc_itype']
    enc_ftype = encoder['enc_ftype']
    enc_btype = encoder['enc_btype']

    return _encode_X_torch(node_dict, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)


def encode_edge(g, encoder):
    edge_dict = _encode_edge_dict(g)
    enc_ptype_edge = encoder['enc_ptype_edge']
    enc_ftype_edge = encoder['enc_ftype_edge']

    return _encode_edge_torch(edge_dict, enc_ftype_edge, enc_ptype_edge)

if __name__ == '__main__':
    data_list = []
    all_kernel = MACHSUITE_KERNEL + poly_KERNEL
    for benchmark_id, kernel_name in enumerate(all_kernel):
        if kernel_name in MACHSUITE_KERNEL:
            dataset = 'machsuite'
        else:
            dataset = 'poly'
        path_graph = join(get_root_path(), 'dse_database', 'programl', dataset, 'processed')
        gexf_file = []
        for f in glob(path_graph + "/*"):
            if f.endswith('.gexf') and kernel_name in f:
                if f[len(path_graph) + len(kernel_name) + 1] == '_':
                    gexf_file.append(f)
        graph_path = join(path_graph, gexf_file[0])
        g = nx.read_gexf(graph_path)
        encoder = load(f'{get_root_path()}/save_models_and_data/encoders.klepto')
        X = encode_node(g, encoder)
        edge_attr = encode_edge(g, encoder)
        edge_index = create_edge_index(g)
        data = Data(
            x=X,
            edge_index=edge_index,
            edge_attr=edge_attr,
            benchmark_id=benchmark_id,
            kernel_name=kernel_name
        )
        torch.save(data, f'{get_root_path()}/SoberDSE/Data/inference/{kernel_name}.pt')

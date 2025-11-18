from config import FLAGS
from train import train_main, inference
from saver import saver
import os.path as osp
from utils import get_root_path
from os.path import join
import config
from programl_data import get_data_list, MyOwnDataset
import time
from dse import ExhaustiveExplorer, NSGAIIExplorer, SAExplorer, MOEDAExplorer, \
ACOExplorer, PSOExplorer, LatticeExplorer, PGExplorer, QLMOEAExplorer, HGBOExplorer, ACExplorer


TARGETS = config.TARGETS
MACHSUITE_KERNEL = config.MACHSUITE_KERNEL
poly_KERNEL = config.poly_KERNEL

if __name__ == '__main__':
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'COLORS-3')
    if not FLAGS.force_regen or FLAGS.subtask == 'dse':
        dataset = MyOwnDataset()
    else:
        pragma_dim = 0
        dataset, pragma_dim = get_data_list()

    if FLAGS.subtask == 'inference':
        inference(dataset)
    elif FLAGS.subtask == 'dse':
        for dataset in ['machsuite','poly']:
            path = join(get_root_path(), 'dse_database', dataset, 'config')
            path_graph = join(get_root_path(), 'dse_database', 'programl', dataset, 'processed')
            # KERNELS = ['doitgen', 'mvt']
            start_time = time.time()
            dict_time = {}
            if dataset == 'machsuite':
                KERNELS = MACHSUITE_KERNEL
            elif dataset == 'poly':
                KERNELS = poly_KERNEL
            else:
                raise NotImplementedError()
            kernel_design_space = {}
            for kernel in KERNELS:
                # if 'md' not in kernel:
                #     continue
                # if kernel in dse_unseen_kernel:
                start_time1 = time.time()
                saver.info('#################################################################')
                saver.info(f'Starting DSE for {kernel}')
                # if FLAGS.explorer == 'Exhastive':
                #     ee = ExhaustiveExplorer(path, kernel, path_graph, run_dse = True)
                #     kernel_design_space[kernel] = ee.ds_size
                # elif FLAGS.explorer == 'NSGA-II':
                #     NSGAIIExplorer(path, kernel, path_graph, run_dse = True)
                # elif FLAGS.explorer == 'SA':
                #     SAExplorer(path, kernel, path_graph, run_dse = True)
                # elif FLAGS.explorer == 'ACO':
                #     ACOExplorer(path, kernel, path_graph, run_dse = True)
                # elif FLAGS.explorer == 'PSO':
                #     PSOExplorer(path, kernel, path_graph, run_dse = True)
                # elif FLAGS.explorer == 'lattice':
                #     LatticeExplorer(path, kernel, path_graph, run_dse = True)
                # elif FLAGS.explorer == 'AC':
                #     ACExplorer(path, kernel, path_graph, run_dse = True)
                # elif FLAGS.explorer == 'PG':
                #     PGExplorer(path, kernel, path_graph, run_dse = True)
                # elif FLAGS.explorer == 'QLMOEA':
                #     QLMOEAExplorer(path, kernel, path_graph, run_dse = True)
                # elif FLAGS.explorer == 'MOEDA':
                #     MOEDAExplorer(path, kernel, path_graph, run_dse = True)
                # elif FLAGS.explorer == 'APDSE':
                #     APDSEExplorer(path, kernel, path_graph, run_dse = True)
                # ExhaustiveExplorer(path, kernel, path_graph, run_dse=True)
                # NSGAIIExplorer(path, kernel, path_graph, run_dse=True)
                # SAExplorer(path, kernel, path_graph, run_dse=True)
                # ACOExplorer(path, kernel, path_graph, run_dse=True)
                # PSOExplorer(path, kernel, path_graph, run_dse=True)
                # LatticeExplorer(path, kernel, path_graph, run_dse=True)
                # ACExplorer(path, kernel, path_graph, run_dse=True)
                # PGExplorer(path, kernel, path_graph, run_dse=True)
                # QLMOEAExplorer(path, kernel, path_graph, run_dse=True)
                # MOEDAExplorer(path, kernel, path_graph, run_dse=True)
                HGBOExplorer(path, kernel, path_graph, run_dse=True)
                saver.info('#################################################################')
                saver.info(f'')
                end_time1 = time.time()
                dict_time[kernel] = end_time1 - start_time1
            end_time = time.time()
            print(f'Runtime {end_time - start_time}')
            print(dict_time)
            print(kernel_design_space)
    else:
        train_main(dataset, 0)


    saver.close()

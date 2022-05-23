import functools

from numpy.core.function_base import logspace
import hparam_parser
import config
import os
import pprint
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import pickle

import torch
import torch.backends.cudnn as cudnn
torch.manual_seed(0)
cudnn.deterministic = True
cudnn.benchmark = False


HPARAM_GRID = {
    'local_lr': [0.03, 0.01, 0.003, 0.001],
}

def recursive_grid_search(grid, base_func, selected_values=None, grid_index=0, counter=0):
    # A recursive function that explores all the values in a grid and call the base_func
    if selected_values is None:
        selected_values = {}

    if grid_index < len(grid):
        k, values = grid[grid_index]
        for v in values:
            selected_values[k] = v
            counter = recursive_grid_search(grid, base_func, selected_values, grid_index + 1, counter)
        return counter
    else:
        base_func(selected_values, counter)
        return counter + 1

def update_run_id(hparam, config_dict, summary_writer, run_id):
    # Update the run id in the hparam dictionary, and the save_dir paramter
    hparam['run_id'] = run_id
    config_dict.update(hparam)
    config_dict['save_dir'] = os.path.join(summary_writer.log_dir, f"run_{run_id}")

def run_hparam(hparam, counter, config_dict, eval_dict, summary_writer, search_record_path):
    run_id = str(counter)
    update_run_id(hparam, config_dict, summary_writer, run_id)
    
    print(f"Running experiment {run_id} with hparam:", hparam)
    if search_record_path is not None:
        # Read the search record file which records which experiment has been started and stored where
        with open(search_record_path, 'rb') as f:
            search_record = pickle.load(f)
        
        # Create a fingerprint for the current config for matching
        config_str = ','.join(str(k) + ":" + str(v) for k, v in sorted(list(config_dict.items()), key=lambda item: str(item[0])) if k != 'run_id' and k != 'save_dir')
        
        if config_str in search_record and search_record[config_str]['status'] == 'finished':
            print(f"Configuration already searched, skipping")
            return

        # If the experiment was started but not finished, remap the run_id and the save_dir to the previous run to allow resumption of training
        if config_str in search_record and search_record[config_str]['status'] == 'started':
            print(f"Remapping experiment {run_id} to {search_record[config_str]['run_id']}")
            run_id = search_record[config_str]['run_id']
            update_run_id(hparam, config_dict, summary_writer, run_id)
        else:
            # Prevent name collision with previous runs, by appending 'n' to the save_dir
            path_appendix = ''
            while os.path.exists(config_dict['save_dir']):
                path_appendix += 'n'
                update_run_id(hparam, config_dict, summary_writer, run_id + path_appendix)
            run_id = run_id + path_appendix

            if len(path_appendix) > 0:
                print(f"New run_id for avoiding Collision: {run_id}")

        search_record[config_str] = {'status': 'started', 'run_id': run_id}
        with open(search_record_path, 'wb') as f:
            pickle.dump(search_record, f)

    
    eval_results = hparam_parser.run(config_dict, eval_dict)

    summary_writer.add_hparams(
        hparam_dict=hparam,
        metric_dict=eval_results
    )

    if search_record_path is not None:
        search_record[config_str] = {'status': 'finished', 'run_id': run_id, 'results': eval_results}
        with open(search_record_path, 'wb') as f:
            pickle.dump(search_record, f)

if __name__ == "__main__":

    parser = hparam_parser.get_parser()
    args = parser.parse_args()
    config_dict = config.get_config_dict()
    eval_dict = config.get_eval_dict()
    config_dict, eval_dict = hparam_parser.update_configs(args, config_dict, eval_dict)


    hparam_grid_list = list(HPARAM_GRID.items())
    log_dir = config_dict['save_dir']

    with SummaryWriter(log_dir=log_dir) as w:
        pp = pprint.PrettyPrinter(width=30)
        with open(os.path.join(log_dir, "hparam_grid.txt"), 'w') as f:
            f.write(pp.pformat(HPARAM_GRID))

        # Create the search record file
        search_record_path = os.path.join(log_dir, "hparam_record.pkl")
        if config_dict['force_restart_hparam'] or not os.path.exists(search_record_path):
            with open(search_record_path, 'wb') as f:
                pickle.dump({}, f)
        
        # If force restarting hparam search, force restart all training
        if config_dict['force_restart_hparam']:
            config_dict['force_restart_training'] = True

        base_func = functools.partial(run_hparam, config_dict=config_dict, eval_dict=eval_dict, summary_writer=w, search_record_path=search_record_path)
        recursive_grid_search(hparam_grid_list, base_func)

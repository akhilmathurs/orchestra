from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import OrderedDict
import warnings
import argparse
import shutil
import types

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import utils
from models import create_backbone, simclr, simsiam, byol, specloss, rotpred, orchestra
import os, shutil
import config
import numpy as np
import pickle as pkl

from myfedavg import MyFedAvg
import client
import functools

cudnn.deterministic = True
cudnn.benchmark = False


def get_eval_fn(config_dict, net, device, global_acc_dict):
    """Return an evaluation function for server-side evaluation."""
    _, memloader, testloader = utils.load_data(config_dict, client_id=-1, bsize=256)

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        params_dict = zip(net.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.Tensor(np.array([v])) if (v.shape == ()) else torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=False)
        
        if(config_dict["train_mode"]=="sup"):
            loss, accuracy = utils.test(net, testloader, device=device) 
            global_acc_dict.append(accuracy)
            return loss, {"accuracy": accuracy}
        else:
            accuracy = utils.knn_monitor(net.backbone, memloader, testloader, device=device)
            global_acc_dict.append(accuracy)
            return -10., {"accuracy": accuracy}
        
    return evaluate

def server_run(config_dict):
    if config_dict["virtualize"]:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=config_dict["CUDA_VISIBLE_DEVICES"]

        vram_per_client_MB = config_dict["client_vram"]
        
        gpu_per_client = 0.0
        if torch.cuda.is_available():
            vram_per_device = float(torch.cuda.get_device_properties(0).total_memory) / (1024**2)
            vram_per_device -= vram_per_client_MB # Allow memory for server
            specified_gpu_per_client = vram_per_client_MB / vram_per_device

            num_gpu = torch.cuda.device_count()
            concurrent_clients = max(1, int(np.ceil(config_dict["num_clients"] * config_dict["fraction_fit"])))
            max_utilization_client_per_gpu = -(concurrent_clients // -num_gpu)
            max_utilization_gpu_per_client = 1.0 / max_utilization_client_per_gpu

            gpu_per_client = max(specified_gpu_per_client, max_utilization_gpu_per_client)

            print(f"GPU Allocation for Simulation:\n\tnum_gpu: {num_gpu}\n\tGPU memory per device: {vram_per_device} MB\n\tConcurrent clinets: {concurrent_clients}\n\tGPU percentage per client: {gpu_per_client:.2%}\n\tMemory_per_client: {gpu_per_client * vram_per_device:.2f} MB\n\tmaximum_clients_per_gpu: {int(1/gpu_per_client)}\n\tmaximum_concurrent_clients: {int(1/gpu_per_client) * num_gpu}\n\tSpecified gpu percentage {specified_gpu_per_client:.2%}, memory {vram_per_client_MB} MB" +
                (" (Overriden because more resources available)" if specified_gpu_per_client < max_utilization_gpu_per_client else "")            
            )
        else:
            print(f"GPU not available, GPU percentage per client: {gpu_per_client:.2%}")

        client_resources = {'num_cpus': 1, 'num_gpus': gpu_per_client}

    device = torch.device(config_dict["main_device"] if torch.cuda.is_available() else "cpu")
    global_acc_dict = []

    os.makedirs(f'{config_dict["save_dir"]}/saved_models', exist_ok=True)
    
    ##### Create model
    n_classes = 10 if (config_dict["dataset"]=="CIFAR10") else 100

    # Define model
    if(config_dict["train_mode"]=="sup"):
        net = create_backbone(name=config_dict["model_class"], num_classes=n_classes, block=config_dict["block"])
        net = net.to(device)
    else:
        if(config_dict["train_mode"]=="simclr"):
            net = simclr(config_dict=config_dict, bbone_arch=config_dict["model_class"]) 
        elif(config_dict["train_mode"]=="simsiam"):
            net = simsiam(config_dict=config_dict, bbone_arch=config_dict["model_class"])
        elif(config_dict["train_mode"]=="byol"):
            net = byol(config_dict=config_dict, bbone_arch=config_dict["model_class"]) 
        elif(config_dict["train_mode"]=="specloss"):
            net = specloss(config_dict=config_dict, bbone_arch=config_dict["model_class"]) 
        elif(config_dict["train_mode"]=="rotpred"):
            net = rotpred(config_dict=config_dict, bbone_arch=config_dict["model_class"])
        elif(config_dict["train_mode"]=="orchestra"):
            net = orchestra(config_dict=config_dict, bbone_arch=config_dict["model_class"])
        else:
            raise Exception
        net = net.to(device)

    # Add 1 to number of rounds if orchestra is used (because first round is spent setting up the centroids)
    n_rounds = config_dict['num_rounds']
    if('orchestra' in config_dict['train_mode']):
        n_rounds = config_dict['num_rounds'] + 1 

    # Load server model and stats from cache if resuming training 
    server_cache_path = f'{config_dict["save_dir"]}/saved_models/{config_dict["dataset"]}_server_cache.pkl'
    trained_rounds = 0
    if not config_dict["force_restart_training"]:
        if os.path.exists(server_cache_path):
            print("Found exisiting server cache, resuming training")
            with open(server_cache_path, 'rb') as f:
                server_cache = pkl.load(f)
                # print("Cache", server_cache)
            trained_rounds = server_cache['rnd']
            global_acc_dict = server_cache['global_acc_dict']
            parameters = fl.common.parameters_to_weights(server_cache['parameters'])
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(np.array([v])) if (v.shape == ()) else torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=False)
            print("Cache", trained_rounds, global_acc_dict, len(parameters))
        else:
            print("Could not find exisiting server cache, starting new training")

    # Define strategy
    strategy = MyFedAvg(
        fraction_fit=config_dict['fraction_fit'],
        fraction_eval=config_dict['fraction_eval'],
        min_eval_clients=config_dict['num_clients'],
        initial_parameters=fl.common.weights_to_parameters([val.cpu().numpy() for _, val in net.state_dict().items()]),
        eta=1.,
        eval_fn=get_eval_fn(config_dict, net, device, global_acc_dict),
        config_dict=config_dict
    )

    # Do not skip first evaluation if force restart or starting from scratch
    if config_dict["force_restart_training"] or trained_rounds == 0:
        strategy._custom_skip_round = False
    else:
        strategy._custom_skip_round = True
            
    # Change behaviours of configure_fit, evaluate, configure_evaluate of the strategy in order to allow resuming training
    strategy.original_configure_fit = strategy.configure_fit
    def configure_fit_patch(self, rnd, parameters, client_manager):
        # Check whether the current round should be skipped from cache
        if rnd <= trained_rounds:
            self._custom_skip_round = True # Setting an attribute of the strategy to change behaviours of following steps (evaluate and configure_evaluate)
            print(f"skipping configure_fit at round {rnd} <= trained_epoch {trained_rounds}")
            return None
        else:
            self._custom_skip_round = False
            print(f"configure_fit at round {rnd} ")
        return self.original_configure_fit(rnd, parameters, client_manager)
    strategy.configure_fit = types.MethodType(configure_fit_patch, strategy)

    strategy.original_evaluate = strategy.evaluate
    def evaluate_patch(self, parameters):
        if hasattr(self, '_custom_skip_round') and self._custom_skip_round:
            print("Skipping evaluate")
            return None
        return self.original_evaluate(parameters)
    strategy.evaluate = types.MethodType(evaluate_patch, strategy)

    strategy.original_configure_evaluate = strategy.configure_evaluate
    def configure_evaluate_patch(self, rnd, parameters, client_manager):
        if hasattr(self, '_custom_skip_round') and self._custom_skip_round:
            print(f"Skipping configure_evaluate at round {rnd}")
            return None

        # Save server model and other metadata to a cache file during configure_evaluate
        print("Saving server training cache to", server_cache_path)
        with open(server_cache_path, 'wb') as f:
            pkl.dump({
                'rnd': rnd,
                'parameters': parameters,
                'global_acc_dict': global_acc_dict
            }, f)
        return self.original_configure_evaluate(rnd, parameters, client_manager)
    strategy.configure_evaluate = types.MethodType(configure_evaluate_patch, strategy)

    # Start server
    if config_dict["virtualize"]:
        make_client_func = functools.partial(client.make_client, config_dict=config_dict)

        fl.simulation.start_simulation(
            client_fn=make_client_func,
            num_clients=config_dict["num_clients"],
            client_resources=client_resources , 
            num_rounds=n_rounds,
            strategy=strategy,
            ray_init_args = {
                "ignore_reinit_error": True,
                "include_dashboard": False,
            }
        )
    else:
        fl.server.start_server(
                server_address="[::]:9081",
            config={"num_rounds": n_rounds},
            strategy=strategy,
        )

    # Save model
    state = {'net': net.state_dict()}
    if(config_dict["train_mode"]=="orchestra"):
        model_name = f'{config_dict["train_mode"]}_{config_dict["num_clients"]}_clients_{config_dict["local_bsize"]}_bsize_{config_dict["local_epochs"]}_lepochs_{config_dict["fraction_fit"]}_fit_{config_dict["num_global_clusters"]}_gclusters_{config_dict["num_local_clusters"]}_lclusters_{config_dict["seed"]}_seed'
    elif(config_dict["train_mode"]=="specloss"):
        model_name = f'{config_dict["train_mode"]}_{config_dict["num_clients"]}_clients_{config_dict["local_bsize"]}_bsize_{config_dict["local_epochs"]}_lepochs_{config_dict["fraction_fit"]}_fit_{config_dict["num_global_clusters"]}_specclusters_{config_dict["seed"]}_seed'
    else:
        model_name = f'{config_dict["train_mode"]}_{config_dict["num_clients"]}_clients_{config_dict["local_bsize"]}_bsize_{config_dict["local_epochs"]}_lepochs_{config_dict["fraction_fit"]}_fit_{config_dict["seed"]}_seed'
    torch.save(state, f'{config_dict["save_dir"]}/saved_models/model_{config_dict["dataset"]}_{config_dict["alpha"]}_alpha_'+model_name+'.pth')

    # Save accuracies
    with open(f'{config_dict["save_dir"]}/saved_models/stats_{config_dict["dataset"]}_{config_dict["alpha"]}_alpha_'+model_name+'.pkl', "wb") as f:
        pkl.dump(global_acc_dict, f)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--client_vram", type=int, default=2500)
    args = parser.parse_args()
    config_dict = config.get_config_dict()
    config_dict['client_vram'] = args.client_vram
    torch.manual_seed(config_dict['seed'])
    server_run(config_dict)
    

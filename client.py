from collections import OrderedDict
import argparse
import warnings
import traceback
import os
import time
import gc

import flwr as fl
from numpy.core.fromnumeric import trace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import pickle as pkl
import numpy as np
import utils
from config import get_config_dict
from models import create_backbone, simclr, simsiam, byol, specloss, rotpred, orchestra

warnings.filterwarnings("ignore", category=UserWarning)
cudnn.deterministic = True
cudnn.benchmark = False


##### Train functions
# Supervised training
def sup_train(net, trainloader, epochs, lr, device=None):
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    for _ in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


# SSL trainer
def ssl_train(net, trainloader, epochs, lr, device=None, is_orchestra=False):
    net.train()

    # Optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # First round of Orchestra only performs local clustering, no training, to initialize global centroids
    if((net.rounds_done[0]==0) and is_orchestra): 
        # Initializing Memory
        net.reset_memory(trainloader, device=device)
        net.local_clustering(device=device)
        return -1

    for _ in range(epochs):
        
        for batch_idx, ((data1, data2), labels) in enumerate(trainloader):
            input1 = data1.to(device)
            if(is_orchestra):
                input2, input3, deg_labels = data2[0].to(device), data2[1].to(device), data2[2].to(device)
            else:
                input2, input3, deg_labels = data2.to(device), None, None

            optimizer.zero_grad()
            loss = net(input1, input2, input3, deg_labels)
            loss.backward()
            optimizer.step()

    if(is_orchestra):
        net.local_clustering(device=device)


# Rotation prediction trainer
def rot_train(net, trainloader, epochs, lr, device=None):
    net.train()

    # Optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    for _ in range(epochs):

        for batch_idx, ((input1, angles), labels) in enumerate(trainloader):
            input1, angles = input1.to(device), angles.to(device)
            optimizer.zero_grad()
            loss = net(input1, angles)
            loss.backward()
            optimizer.step()


#### Client definitions
def make_client(cid, device=None, stateless=True, config_dict=None):
    try:
        gc.collect()
        torch.cuda.empty_cache()    
        client_id = int(cid) # cid is of type str when using simulation

        if device is None:
            print("Client {} CUDA_VISIBLE_DEVICES: {}".format(cid, os.environ["CUDA_VISIBLE_DEVICES"]))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_save_path = config_dict["save_dir"]+"/saved_models/"+config_dict["dataset"]+"_client_"+str(client_id)+".pth"

        ##### Create model
        n_classes = 10 if (config_dict["dataset"]=="CIFAR10") else 100

        ##### Load data
        trainloader, memloader, testloader = utils.load_data(config_dict, client_id=client_id, n_clients=config_dict['num_clients'], alpha=config_dict['alpha'],
                                                                bsize=config_dict["local_bsize"], in_simulation=config_dict["virtualize"])


        # Define model; for SSL, projector/predictor will also be needed
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
            net = net.to(device)


        ##### Flower client
        class flclient(fl.client.NumPyClient):
            def get_parameters(self):
                return [val.cpu().numpy() for _, val in net.state_dict().items()]

            def set_parameters(self, parameters):
                params_dict = zip(net.state_dict().keys(), parameters)
                if(config_dict['stateful_client']):
                    state_dict = OrderedDict({k: torch.Tensor(np.array([v])) if (v.shape == ()) else torch.Tensor(v) for k, v in params_dict if ('mem_projections' not in k and 'target_' not in k)})
                else:
                    state_dict = OrderedDict({k: torch.Tensor(np.array([v])) if (v.shape == ()) else torch.Tensor(v) for k, v in params_dict if ('mem_projections' not in k)})

                net.load_state_dict(state_dict, strict=False)

            def fit(self, parameters, config):
                try:
                    self.set_parameters(parameters)

                    # Supervised training
                    if(config_dict["train_mode"]=="sup"):
                        sup_train(net, trainloader, epochs=config_dict["local_epochs"], lr=config_dict["local_lr"], device=device)

                    # SSL training
                    else:
                        if(config_dict['train_mode']=='rotpred'):
                            rot_train(net, trainloader, epochs=config_dict["local_epochs"], lr=config_dict['local_lr'], device=device)
                        else:
                            ssl_train(net, trainloader, epochs=config_dict["local_epochs"], lr=config_dict['local_lr'], device=device, is_orchestra=config_dict["train_mode"]=="orchestra")

                    return self.get_parameters(), len(trainloader), {}
                except Exception as e:
                    print(f"Client {cid} - Exception in client fit {e}")
                    print(f"Client {cid}", traceback.format_exc())

            def evaluate(self, parameters, config):
                self.set_parameters(parameters)
                if(config_dict["train_mode"]=="sup"):
                    loss, accuracy = utils.test(net, testloader, device=device, verbose=False)
                    return float(loss), len(testloader), {"accuracy": float(accuracy)}
                else:
                    accuracy = utils.knn_monitor(net.backbone, memloader, testloader, verbose=False, device=device)
                    return float(0), len(testloader), {"accuracy": float(accuracy)}

            def save_net(self):
                ##### Save local model
                state = {'net': net.state_dict()}
                torch.save(state, model_save_path)
                print(f"Client: {client_id} Saving network to {model_save_path}")
            
        gc.collect()
        torch.cuda.empty_cache()
        return flclient()
    except Exception as e:
        print(f"Client {cid} - Exception in make_client {e}")
        print(f"Client {cid}", traceback.format_exc())

##### Federation of the pipeline with Flower
def main(config_dict):
    """Create model, load data, define Flower client, start Flower client."""
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--client_id", type=int, default=0)
    args = parser.parse_args()

    # device = torch.device(config_dict["main_device"] if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:"+str(args.client_id%8) if torch.cuda.is_available() else "cpu")

    local_client = make_client(args.client_id, device=device, stateless=True, config_dict=config_dict)

    ##### Start client
    fl.client.start_numpy_client("[::]:9081", client=local_client)

    local_client.save_net()

if __name__ == "__main__":
    config_dict = get_config_dict()
    torch.manual_seed(config_dict['seed'])

    main(config_dict)
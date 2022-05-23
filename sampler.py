import numpy as np
import argparse
import torch
from torchvision import datasets, transforms
import pickle as pkl
import os, shutil
import utils

parser = argparse.ArgumentParser(description="Sample data for clients")
parser.add_argument("--dataset", default="CIFAR10", choices=["CIFAR10", "CIFAR100"])
parser.add_argument("--n_clients", type=int, default=10)
parser.add_argument("--alpha", type=float, default=1e5, choices=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
parser.add_argument("--use_IID", type=str, default='False', choices=['True', 'False'])
parser.add_argument("--use_balance", type=str, default='True', choices=['True', 'False'])
parser.add_argument("--data_dir", default="./data/")

args = parser.parse_args()
args.use_IID = (args.use_IID=='True')
args.use_balance = (args.use_balance=='True')

torch.manual_seed(0)
np.random.seed(0)

os.makedirs(f'{args.data_dir}/{args.n_clients}/{args.alpha}/{args.dataset}', exist_ok=True)
os.makedirs(f'{args.data_dir}/{args.n_clients}/{args.alpha}/{args.dataset}/train', exist_ok=True)
os.makedirs(f'{args.data_dir}/{args.n_clients}/{args.alpha}/{args.dataset}/test', exist_ok=True)

##### Print setup to confirm things are correct
print("\nSampling configuration:")
print("\tDataset:", args.dataset)
print("\tNumber of clients:", args.n_clients)
print("\tDistribute IID:", args.use_IID)
print("\tCreate balanced partitions:", args.use_balance)
print("\tWriting data at this location: ", args.data_dir + "/" + str(args.n_clients))
if(not args.use_IID):
    print("\tAlpha for Dirichlet distribution:", args.alpha)
print("\n")

##### Determine number of samples in dataset
if(args.dataset=="CIFAR10"):
    n_classes = 10
    train_data = datasets.CIFAR10(f'{args.data_dir}/dataset/CIFAR10', train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))
    test_data = datasets.CIFAR10(f'{args.data_dir}/dataset/CIFAR10', train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))
elif(args.dataset=="CIFAR100"):
    n_classes = 100
    train_data = datasets.CIFAR100(f'{args.data_dir}/dataset/CIFAR100', train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))
    test_data = datasets.CIFAR100(f'{args.data_dir}/dataset/CIFAR100', train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))
else:
    raise Exception("Dataset not recognized")
n_samples_train = len(train_data)
n_samples_test = len(test_data)


##### Determine locations of different classes
all_ids_train = np.array(train_data.targets)
class_ids_train = {class_num: np.where(all_ids_train == class_num)[0] for class_num in range(n_classes)}
all_ids_test = np.array(test_data.targets)
class_ids_test = {class_num: np.where(all_ids_test == class_num)[0] for class_num in range(n_classes)}


##### Determine distribution over classes to be assigned per client
# Returns n_clients x n_classes matrix
n_clients = args.n_clients
if(args.use_IID):
    args.alpha = 1e5
dist_of_client = np.random.dirichlet(np.repeat(args.alpha, n_clients), size=n_classes).transpose()
dist_of_client /= dist_of_client.sum()

#### Run OT if using balanced partitioning
if(args.use_balance):
    for i in range(100):
        s0 = dist_of_client.sum(axis=0, keepdims=True)
        s1 = dist_of_client.sum(axis=1, keepdims=True)
        dist_of_client /= s0
        dist_of_client /= s1

##### Allocate number of samples per class to each client based on distribution
samples_per_class_train = (np.floor(dist_of_client * n_samples_train))
samples_per_class_test = (np.floor(dist_of_client * n_samples_test))

start_ids_train = np.zeros((n_clients+1, n_classes), dtype=np.int32)
start_ids_test = np.zeros((n_clients+1, n_classes), dtype=np.int32)
for i in range(0, n_clients):
    start_ids_train[i+1] = start_ids_train[i] + samples_per_class_train[i]
    start_ids_test[i+1] = start_ids_test[i] + samples_per_class_test[i]


# Sanity checks
print("\nSanity checks:")
print("\tSum of dist. of classes over clients: {}".format(dist_of_client.sum(axis=0)))
print("\tSum of dist. of clients over classes: {}".format(dist_of_client.sum(axis=1)))
print("\tTotal trainset size: {}".format(samples_per_class_train.sum()))
print("\tTotal testset size: {}".format(samples_per_class_test.sum()))


##### Save IDs
# Train
client_ids = {client_num: {} for client_num in range(n_clients)}
for client_num in range(n_clients):
    l = np.array([], dtype=np.int32)
    for class_num in range(n_classes):
        start, end = start_ids_train[client_num, class_num], start_ids_train[client_num+1, class_num]
        l = np.concatenate((l, class_ids_train[class_num][start:end].tolist())).astype(np.int32)
    client_ids[client_num] = l


print("\nDistribution over classes:")
for client_num in range(n_clients):
    with open(f"{args.data_dir}/{args.n_clients}/{args.alpha}/{args.dataset}/train/"+args.dataset+"_"+str(client_num)+'.pkl', 'wb') as f:
        pkl.dump(client_ids[client_num], f)
    print("\tClient {cnum}: \n \t\t Train: {cdisttrain} \n \t\t Total: {traintotal} \n \t\t Test: {cdisttest} \n \t\t Total: {testtotal}".format(
        cnum=client_num, cdisttrain=samples_per_class_train[client_num], cdisttest=samples_per_class_test[client_num], 
        traintotal=samples_per_class_train[client_num].sum(), testtotal=samples_per_class_test[client_num].sum()))

# Test
client_ids = {client_num: {} for client_num in range(n_clients)}
for client_num in range(n_clients):
    l = np.array([], dtype=np.int32)
    for class_num in range(n_classes):
        start, end = start_ids_test[client_num, class_num], start_ids_test[client_num+1, class_num]
        l = np.concatenate((l, class_ids_test[class_num][start:end].tolist())).astype(np.int32)
    client_ids[client_num] = l

for client_num in range(n_clients):
    with open(f"{args.data_dir}/{args.n_clients}/{args.alpha}/{args.dataset}/test/"+args.dataset+"_"+str(client_num)+'.pkl', 'wb') as f:
        pkl.dump(client_ids[client_num], f)

from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import OrderedDict
import pickle
import flwr as fl

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100
import pickle as pkl
import numpy as np
import sys
import time
from tqdm import tqdm
from PIL import Image, ImageOps, ImageFilter
import random

torch.manual_seed(0)
cudnn.deterministic = True
cudnn.benchmark = False


######### Client Dataset class #########
class clientDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.classes = dataset.classes
        self.targets = np.array(dataset.targets)[idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def get_dataset_subset(dataset, subset_proportion=1, force_class_balanced=False, seed=0):
    if subset_proportion >= 1:
        return dataset

    targets_np = np.array(dataset.targets)
    unique_classes, class_counts = np.unique(targets_np, return_counts=True)

    # Calculate the number of samples per class if class balancing is needed
    balanced_target_per_class = math.ceil(len(targets_np) * subset_proportion / len(unique_classes))

    rng = np.random.default_rng(seed)
    extracted_indices = []
    for c in unique_classes:
        # Find the indices corresponding to the current class
        class_indices = np.where(targets_np == c)[0]
        rng.shuffle(class_indices)
        if force_class_balanced:
            # Extract as many samples as possible when class balancing is needed
            num_samples_extrated = min(balanced_target_per_class, len(class_indices))
        else:
            # Follow the original distribution of the dataset otherwise
            num_samples_extrated = math.ceil(len(class_indices) * subset_proportion)
        extracted_indices.append(class_indices[:num_samples_extrated])
    extracted_indices = np.sort(np.concatenate(extracted_indices))

    subset_dataset = clientDataset(dataset, extracted_indices)
    return subset_dataset


########## Custom Pickled Dataset class #########

class PickledVisionDataset(torchvision.datasets.VisionDataset):
    def __init__(self, pickled_file_path, transform=None, target_transform=None):
        super(PickledVisionDataset, self).__init__(pickled_file_path, transform=transform,
                                      target_transform=target_transform)
        self.pickled_file_path = pickled_file_path
        with open(pickled_file_path, 'rb') as f:
            dataset = pickle.load(f)
        self.data = dataset['x']
        self.targets = dataset['y']
        self.idx_to_class = {}
        if 'idx_to_class' in dataset:
            self.idx_to_class = dataset['idx_to_class']
            self.classes = list(self.idx_to_class.values())

    def get_class(self, target):
        return self.idx_to_class[target]

    def __getitem__(self, index):
        # Adapted from torchvision.datasets.CIFAR10
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


########## Data Augmentations ##########
# We follow augmentations detailed in official implementations, which seem to be tuned to make the corresponding method work best
class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class Solarization(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def image_rot(image, angle):
    image = TF.rotate(image, angle)
    return image


class BaseTransform():
    def __init__(self, is_sup, image_size=32):
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.mode = is_sup

    def __call__(self, x):
        if(self.mode):
            return self.transform(x)
        else:
            x1 = self.transform(x)
            x2 = self.transform(x)
            return x1, x2 


class SimCLRTransform():
    def __init__(self, is_sup, image_size=32):
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.mode = is_sup

    def __call__(self, x):
        if(self.mode):
            return self.transform(x)
        else:
            x1 = self.transform(x)
            x2 = self.transform(x)
            return x1, x2 


class SpecLossTransform():
    def __init__(self, is_sup, image_size=32):
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform_prime = T.Compose([
            T.RandomResizedCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])    

        self.mode = is_sup

    def __call__(self, x):
        if(self.mode):
            return self.transform(x)
        else:
            x1 = self.transform_prime(x)
            x2 = self.transform(x)
            return x1, x2

class BYOLTransform():
    def __init__(self, is_sup, image_size=32):
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=1.0),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform_prime = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.1),
            Solarization(p=0.2),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.mode = is_sup

    def __call__(self, x):
        if(self.mode):
            return self.transform(x)
        else:
            x1 = self.transform(x)
            x2 = self.transform_prime(x)
            return x1, x2 


class RotTransform():
    def __init__(self, is_sup):
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.mode = is_sup

    def __call__(self, x):
        n = random.random()
        angle = 0 if n <= 0.25 else 1 if n <= 0.5 else 2 if n <= 0.75 else 3
        return self.transform(image_rot(x, 90*angle)), angle 


class OrchestraTransform():
    def __init__(self, is_sup, image_size=32):
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform_prime = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.1),
            Solarization(p=0.2),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.mode = is_sup

    def __call__(self, x):
        n = random.random()
        angle = 0 if n <= 0.25 else 1 if n <= 0.5 else 2 if n <= 0.75 else 3
        if(self.mode):
            return self.transform(x)
        else:
            x1 = self.transform(x)
            x2 = self.transform_prime(x)
            x3 = image_rot(self.transform(x), 90*angle)
            return x1, [x2, x3, angle]


######### Dataloaders #########
def load_data(config_dict, client_id=-1, n_clients=50, alpha=1e0, bsize=16, 
              linear_eval=False, hparam_eval=False, in_simulation=False, force_shuffle=False, 
              subset_proportion=1, subset_force_class_balanced=False, subset_seed=0):
    
    da_method = config_dict["da_method"]
    train_mode = config_dict["train_mode"]
    dataset_name = config_dict["dataset"]
    data_dir = config_dict['data_dir']
    
    # Define data augmentations
    if(hparam_eval):
        transform_train = SimCLRTransform(is_sup=False, image_size=32)
    elif(linear_eval):
        transform_train = BaseTransform(is_sup=True, image_size=32)
    elif(da_method=="sup"):
        transform_train = BaseTransform(is_sup=(train_mode=="sup"), image_size=32)
    elif(da_method=="simclr" or da_method=="simsiam"):
        transform_train = SimCLRTransform(is_sup=(train_mode=="sup"), image_size=32)
    elif(da_method=="specloss"):
        transform_train = SpecLossTransform(is_sup=(train_mode=="sup"), image_size=32)
    elif(da_method=="byol"):
        transform_train = BYOLTransform(is_sup=(train_mode=="sup"), image_size=32)
    elif(da_method=="rotpred"):
        transform_train = RotTransform(is_sup=(train_mode=="sup"))
    elif(da_method=="orchestra"):
        transform_train = OrchestraTransform(is_sup=(train_mode=="sup"), image_size=32)

    transform_test = T.Compose([
        T.ToTensor(), 
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load dataset
    if(dataset_name=="CIFAR10"):
        trainset = CIFAR10(f"{data_dir}/dataset/CIFAR10", train=True, download=False, transform=transform_train)
        memset = CIFAR10(f"{data_dir}/dataset/CIFAR10", train=True, download=False, transform=transform_test)
        testset = CIFAR10(f"{data_dir}/dataset/CIFAR10", train=False, download=False, transform=transform_test)
    elif(dataset_name=="CIFAR100"):
        trainset = CIFAR100(f"{data_dir}/dataset/CIFAR100", train=True, download=False, transform=transform_train)
        memset = CIFAR100(f"{data_dir}/dataset/CIFAR100", train=True, download=False, transform=transform_test)
        testset = CIFAR100(f"{data_dir}/dataset/CIFAR100", train=False, download=False, transform=transform_test)
    else:
        raise Exception("Dataset not recognized")

    # Dataloaders for given client
    if(client_id > -1):
        with open(f'{data_dir}/{n_clients}/{alpha}/{dataset_name}/train/' +dataset_name+"_"+str(client_id)+".pkl", "rb") as f:
            train_ids = pkl.load(f).astype(np.int32)
        with open(f'{data_dir}/{n_clients}/{alpha}/{dataset_name}/test/'+dataset_name+"_"+str(client_id)+".pkl", "rb") as f:
            test_ids = pkl.load(f).astype(np.int32)
        # Sanity check
        train_deets, test_deets = np.unique(np.array(trainset.targets)[train_ids], return_counts=True), np.unique(np.array(testset.targets)[test_ids], return_counts=True)

        trainloader = DataLoader(clientDataset(trainset, train_ids), batch_size=bsize, shuffle=True, drop_last=True)            
        memloader = DataLoader(clientDataset(memset, train_ids), batch_size=bsize, shuffle=True, drop_last=True)
        testloader = DataLoader(clientDataset(testset, test_ids), batch_size=bsize, shuffle=False, drop_last=True)

        # Sanity check
        if(not in_simulation):
            print("Client: {c}".format(c=client_id))
            print("Train set details: \n\tClasses: {c} \n\tSamples: {s}".format(c=train_deets[0], s=train_deets[1]))
            print("Test set details: \n\tClasses: {c} \n\tSamples: {s}".format(c=test_deets[0], s=test_deets[1]))
            print("\nTrain set size: {}; Test set size: {} \n".format(len(trainloader.dataset), len(testloader.dataset)))

    else:  # client_id == -1 implies server
        if subset_proportion < 1: # enables semi-supervised training
            trainset = get_dataset_subset(trainset, subset_proportion=subset_proportion, force_class_balanced=subset_force_class_balanced, seed=subset_seed)
            memset = get_dataset_subset(memset, subset_proportion=subset_proportion, force_class_balanced=subset_force_class_balanced, seed=subset_seed)
        trainloader = DataLoader(trainset, batch_size=bsize, shuffle=force_shuffle, num_workers=2, drop_last=True)
        memloader = DataLoader(memset, batch_size=bsize, shuffle=force_shuffle, num_workers=2, drop_last=True)
        testloader = DataLoader(testset, batch_size=bsize, shuffle=force_shuffle, num_workers=2, drop_last=True)
        # Sanity check
        print("\nTrain set size: {}; Test set size: {} \n".format(len(trainloader.dataset), len(testloader.dataset)))

    return trainloader, memloader, testloader


########## Test function ##########
def test(net, testloader, device="cpu", verbose=True):
    net.eval()
    correct, total, test_loss = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    if(verbose):
        print("\n")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if(verbose):
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return test_loss/(batch_idx+1), 100. * (correct / total)


#### The following tools were adapted from https://github.com/PatrickHua/SimSiam
# kNN monitor
def knn_monitor(net, memory_data_loader, test_data_loader, k=200, t=0.1, device="cpu", verbose=True):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    feature_labels = []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=not verbose):
            feature = net(data.to(device))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target.to(device))
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        # feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=device)
        feature_labels = torch.cat(feature_labels, dim=0).contiguous()

        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader, desc='kNN', disable=not verbose)
        for data, target in test_bar:
            data, target = data.to(device), target.to(device)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
    return total_top1 / total_num * 100

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

# LR Scheduler
class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch, constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))
        
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0
    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]
        
        self.iter += 1
        self.current_lr = lr
        return lr
    def get_lr(self):
        return self.current_lr


######### Progress bar #########
term_width = 150 
TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')
    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))
    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)
    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
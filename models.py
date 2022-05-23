import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from math import pi, cos, e
import numpy as np
from collections import OrderedDict
from utils import progress_bar


######### Backbone models #########
#### VGG-13
class BlockVGG(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BlockVGG, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes, affine=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) 
        return out


class VGG(nn.Module):
    def __init__(self, block, num_classes=10, cfg=None):
        super(VGG, self).__init__()
        self.cfg = cfg 
        self.train_sup = (num_classes > 0)

        self.layers = self._make_layers(in_planes=3, block=block)
        self.output_dim = self.cfg[-1]
        if(self.train_sup):
            self.linear = nn.Linear(self.cfg[-1] if isinstance(self.cfg[-1], int) else self.cfg[-1][0], num_classes)

    def _make_layers(self, in_planes, block):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = out.mean(dim=(2,3))
        if(self.train_sup):
            out = self.linear(out)
        return out

def VGGmodel(num_classes=10):
    cfg = [64, (64, 2), 128, (128, 2), 256, (256, 2), 512, (512, 2), 512, 512]
    return VGG(BlockVGG, num_classes=num_classes, cfg=cfg)


#### ResNets
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.use_shortcut = stride != 1 or in_planes != self.expansion*planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)

        self.shortcut_conv = nn.Sequential()
        if self.use_shortcut:
            self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes, affine=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut_conv(x)
        if self.use_shortcut:
            shortcut = self.shortcut_bn(shortcut)
        out += shortcut
        return F.relu(out) 


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.use_shortcut = stride != 1 or in_planes != self.expansion*planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes, affine=True)

        self.shortcut_conv = nn.Sequential()
        if self.use_shortcut:
            self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes, affine=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = F.relu(self.bn2(self.conv2(out))) 
        out = self.bn3(self.conv3(out))
        shortcut = self.shortcut_conv(x)
        if self.use_shortcut:
            shortcut = self.shortcut_bn(shortcut)
        out += shortcut
        return F.relu(out) 


# Model class
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cfg=None):
        super(ResNet, self).__init__()
        self.train_sup = (num_classes > 0)
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.output_dim = 512*block.expansion
        if(self.train_sup):
            self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        if(self.train_sup):
            out = self.linear(out)
        return out

class ResNet_basic(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cfg=None):
        super(ResNet_basic, self).__init__()
        self.train_sup = (num_classes > 0)

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16, affine=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.output_dim = 512*block.expansion
        if(self.train_sup):
            self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        if(self.train_sup):
            out = self.linear(out)
        return out


def get_block(block):
    if(block=="BasicBlock"):
        return BasicBlock
    elif(block=="Bottleneck"):
        return Bottleneck

def ResNet18(num_classes=10, block="BasicBlock"):
    return ResNet(get_block(block), [2,2,2,2], num_classes=num_classes)

def ResNet34(num_classes=10, block="BasicBlock"):
    return ResNet(get_block(block), [3,4,6,3], num_classes=num_classes)

def ResNet56(num_classes=10, block="BasicBlock"):
    return ResNet_basic(get_block(block), [9,9,9], num_classes=num_classes)


### Retrieval function for backbones ###
def create_backbone(name, num_classes=10, block='BasicBlock'):
    if(name == 'VGG'):
        net = VGGmodel(num_classes=num_classes)
    elif(name == 'res18'):
        net = ResNet18(num_classes=num_classes, block=block)
    elif(name == 'res34'):
        net = ResNet34(num_classes=num_classes, block=block)
    elif(name == 'res56'):
        net = ResNet56(num_classes=num_classes, block=block)

    return net


######### SimCLR model class #########
def NT_xentloss(z1, z2, temperature=0.5): # this definition adapted from https://github.com/PatrickHua/SimSiam
    N, Z = z1.shape 
    device = z1.device 
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)

    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)

    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]
    negatives = similarity_matrix[~diag].view(2*N, -1)

    logits = torch.cat([positives, negatives], dim=1) / temperature
    labels = torch.zeros(2*N, device=device, dtype=torch.int64) # scalar label per sample
    loss = F.cross_entropy(logits, labels, reduction='sum')

    return loss / (2 * N)

# Projector 
class projection_MLP_simclr(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=512):
        super(projection_MLP_simclr, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layer1_bn = nn.BatchNorm1d(hidden_dim, affine=True)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.layer2_bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        x = F.relu(self.layer1_bn(self.layer1(x))) 
        x = self.layer2_bn(self.layer2(x)) 
        return x

# SimCLR
class simclr(nn.Module):
    def __init__(self, config_dict, bbone_arch):
        super(simclr, self).__init__()
        self.T = config_dict['main_T']
        self.register_buffer("rounds_done", torch.zeros(1))

        self.backbone = create_backbone(bbone_arch, num_classes=0)
        self.projector = projection_MLP_simclr(self.backbone.output_dim, hidden_dim=512, out_dim=512)

    def forward(self, x1, x2, x3=None, deg_labels=None):
        N = x1.shape[0]
        z1, z2 = self.projector(self.backbone(x1)), self.projector(self.backbone(x2))
        L = NT_xentloss(z1, z2, temperature=self.T)

        return L


######### SimSiam model class #########
class MLPact(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPact, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, affine=True)

    def forward(self, x):
        out = F.relu(self.bn(self.linear(x))) 
        return out

# Projector
class projection_MLP_simsiam(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=512):
        super(projection_MLP_simsiam, self).__init__()
        self.output_dim = out_dim
        self.layer1 = MLPact(in_dim, hidden_dim)
        self.layer2 = MLPact(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.layer3_bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        x = self.layer3_bn(self.layer3(self.layer2(self.layer1(x))))
        return x 

# Predictor 
class prediction_MLP_simsiam(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=512): 
        super(prediction_MLP_simsiam, self).__init__()
        self.layer1 = MLPact(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

# SimSiam
class simsiam(nn.Module):
    def __init__(self, config_dict, bbone_arch, ):
        super(simsiam, self).__init__()
        self.register_buffer("rounds_done", torch.zeros(1))

        self.backbone = create_backbone(bbone_arch, num_classes=0)
        self.projector = projection_MLP_simsiam(self.backbone.output_dim, hidden_dim=256, out_dim=512)

        ### Predictor (should be defined last for divergence aware update)
        self.predictor = prediction_MLP_simsiam(in_dim=self.projector.output_dim, out_dim=self.projector.output_dim)
    
    def forward(self, x1, x2, x3=None, deg_labels=None):
        z1, z2 = self.projector(self.backbone(x1)), self.projector(self.backbone(x2))
        p1, p2 = self.predictor(z1), self.predictor(z2)
        L = - (F.cosine_similarity(p1, z2.detach(), dim=-1).mean() + F.cosine_similarity(p2, z1.detach(), dim=-1).mean()) / 2

        return L


######### BYOL model class #########
# Projector / Predictor MLP class for BYOL
class MLP_BYOL(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=512, is_pred=False):
        super(MLP_BYOL, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim, affine=True)
        self.layer2 = nn.Linear(hidden_dim, in_dim, bias=is_pred)

    def forward(self, x):
        x = self.layer2(F.relu(self.bn1(self.linear1(x)))) 
        return x 

# BYOL
class byol(nn.Module):
    def __init__(self, config_dict, bbone_arch):
        super(byol, self).__init__()
        self.ema_value = config_dict['ema_value']
        self.register_buffer("rounds_done", torch.zeros(1))

        # Online model
        self.backbone = create_backbone(bbone_arch, num_classes=0)
        self.projector = MLP_BYOL(in_dim=self.backbone.output_dim, hidden_dim=1024, out_dim=512)

        # Target model
        self.target_backbone = create_backbone(bbone_arch, num_classes=0)

        for param_base, param_target in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            param_target.data.copy_(param_base.data)  # initialize
            param_target.requires_grad = False  # not update by gradient

        self.target_projector = MLP_BYOL(in_dim=self.backbone.output_dim, hidden_dim=1024, out_dim=512)

        for param_base, param_target in zip(self.projector.parameters(), self.target_projector.parameters()):
            param_target.data.copy_(param_base.data)  # initialize
            param_target.requires_grad = False  # not update by gradient

        ### Predictor (should be defined last for divergence aware update)
        self.predictor = MLP_BYOL(in_dim=512, hidden_dim=1024, out_dim=512, is_pred=True)

    @torch.no_grad()
    def update_target(self):
        tau = self.ema_value
        for online, target in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            target.data = (1 - tau) * target.data + (tau) * online.data
        for online, target in zip(self.projector.parameters(), self.target_projector.parameters()):
            target.data = (1 - tau) * target.data + (tau) * online.data

    def forward(self, x1, x2, x3=None, deg_labels=None):
        p1, p2 = self.projector(self.backbone(x1)), self.projector(self.backbone(x2))
        with torch.no_grad():
            self.update_target()
            z1, z2 = self.target_projector(self.target_backbone(x1)), self.target_projector(self.target_backbone(x2))         
        p1, p2 = self.predictor(p1), self.predictor(p2)

        L = - (F.cosine_similarity(p1, z2, dim=-1).mean() + F.cosine_similarity(p2, z1, dim=-1).mean()) / 2

        return L


######### SpecLoss model class #########
# Projector
class projection_MLP_specloss(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=512):
        super(projection_MLP_specloss, self).__init__()
        self.layer1 = MLPact(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.layer2_bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        x = self.layer2_bn(self.layer2(self.layer1(x)))
        return x 

# Spectral contrastive loss
def l_specloss(z1, z2): 
    N, Z = z1.shape 
    device = z1.device 
    similarity_matrix = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)

    pos = torch.diag(similarity_matrix)
    diag = torch.eye(N, dtype=torch.bool, device=device)
    negatives = similarity_matrix[~diag]

    loss = - (2 * pos.sum() / N) + (negatives.pow(2).sum() / (N*(N-1)))
    return loss 

# SpecLoss model
class specloss(nn.Module):
    def __init__(self, config_dict, bbone_arch):
        super(specloss, self).__init__()
        self.register_buffer("rounds_done", torch.zeros(1))

        self.backbone = create_backbone(bbone_arch, num_classes=0)
        self.projector = projection_MLP_specloss(self.backbone.output_dim, hidden_dim=512, out_dim=512)
        self.centroids = nn.Linear(512, config_dict['num_global_clusters']) # Number of clusters extracted via spectral clustering 
        # Note that specloss sets centroids equal to number of classes--this info is not technically available, so we use same clusters as Orchestra

    def forward(self, x1, x2, x3=None, deg_labels=None):
        z1, z2 = F.normalize(self.projector(self.backbone(x1)), dim=1), F.normalize(self.projector(self.backbone(x2)), dim=1)
        L = l_specloss(self.centroids(z1), self.centroids(z2))

        return L


############ Rotation prediction model class ############
# Projector 
class projection_MLP_rotpred(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=512):
        super(projection_MLP_rotpred, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layer1_bn = nn.BatchNorm1d(hidden_dim, affine=True)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.layer2_bn = nn.BatchNorm1d(out_dim, affine=False) 

    def forward(self, x):
        x = F.relu(self.layer1_bn(self.layer1(x))) 
        x = self.layer2_bn(self.layer2(x)) 
        return x

# Rotation prediction
class rotpred(nn.Module):
    def __init__(self, config_dict, bbone_arch):
        super(rotpred, self).__init__()
        self.register_buffer("rounds_done", torch.zeros(1))

        self.backbone = create_backbone(bbone_arch, num_classes=0)
        self.projector = projection_MLP_rotpred(self.backbone.output_dim)
        self.rotation_pred = nn.Linear(512, 4)

    def forward(self, images, angles):
        N = images.shape[0]
        r_preds = self.rotation_pred(self.projector(self.backbone(images)))
        L = F.cross_entropy(r_preds, angles) 
        return L, 0, 0


############ Orchestra model class ############
# Sinkhorn Knopp 
def sknopp(cZ, lamd=25, max_iters=100):
    with torch.no_grad():
        N_samples, N_centroids = cZ.shape # cZ is [N_samples, N_centroids]
        probs = F.softmax(cZ * lamd, dim=1).T # probs should be [N_centroids, N_samples]

        r = torch.ones((N_centroids, 1), device=probs.device) / N_centroids # desired row sum vector
        c = torch.ones((N_samples, 1), device=probs.device) / N_samples # desired col sum vector

        inv_N_centroids = 1. / N_centroids
        inv_N_samples = 1. / N_samples

        err = 1e3
        for it in range(max_iters):
            r = inv_N_centroids / (probs @ c)  # (N_centroids x N_samples) @ (N_samples, 1) = N_centroids x 1
            c_new = inv_N_samples / (r.T @ probs).T  # ((1, N_centroids) @ (N_centroids x N_samples)).t() = N_samples x 1
            if it % 10 == 0:
                err = torch.nansum(torch.abs(c / c_new - 1))
            c = c_new
            if (err < 1e-2):
                break

        # inplace calculations. 
        probs *= c.squeeze()
        probs = probs.T # [N_samples, N_centroids]
        probs *= r.squeeze()

        return probs * N_samples # Soft assignments


# Projector 
class projection_MLP_orchestra(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=512):
        super(projection_MLP_orchestra, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layer1_bn = nn.BatchNorm1d(hidden_dim, affine=True)
        self.layer2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.layer2_bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        x = F.relu(self.layer1_bn(self.layer1(x))) 
        x = self.layer2_bn(self.layer2(x)) 
        return x

# Model class
class orchestra(nn.Module):
    def __init__(self, config_dict, bbone_arch):
        super(orchestra, self).__init__()

        # Setup arguments
        self.N_local = config_dict['num_local_clusters'] # Number of local clusters
        self.N_centroids = config_dict['num_global_clusters'] # Number of centroids 
        self.m_size = config_dict['cluster_m_size'] # Memory size for projector representations 
        self.T = config_dict['main_T']

        self.ema_value = config_dict['ema_value']

        self.register_buffer('rounds_done', torch.zeros(1))

        ### Online Model
        self.backbone = create_backbone(bbone_arch, num_classes=0)
        self.projector = projection_MLP_orchestra(in_dim=self.backbone.output_dim, out_dim=512)

        ### Target model
        self.target_backbone = create_backbone(bbone_arch, num_classes=0)

        for param_base, param_target in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            param_target.data.copy_(param_base.data)  # initialize
            param_target.requires_grad = False  # not updated by gradient

        self.target_projector = projection_MLP_orchestra(in_dim=self.backbone.output_dim, out_dim=512)

        for param_base, param_target in zip(self.projector.parameters(), self.target_projector.parameters()):
            param_target.data.copy_(param_base.data)  # initialize
            param_target.requires_grad = False  # not updated by gradient

        ### Degeneracy regularization layer (implemented via rotation prediction)
        self.deg_layer = nn.Linear(512, 4)

        ### Centroids [D, N_centroids] and projection memories [D, m_size]
        self.mem_projections = nn.Linear(self.m_size, 512, bias=False)
        self.centroids = nn.Linear(512, self.N_centroids, bias=False) # must be defined second last
        self.local_centroids = nn.Linear(512, self.N_local, bias=False) # must be defined last


    @torch.no_grad()
    def reset_memory(self, data, device='cuda:7'):
        self.train()

        # Save BN parameters to ensure they are not changed when initializing memory
        reset_dict = OrderedDict({k: torch.Tensor(np.array([v.cpu().numpy()])) if (v.shape == ()) else torch.Tensor(v.cpu().numpy()) for k, v in self.state_dict().items() if 'bn' in k})

        # generate feature bank
        proj_bank = []
        n_samples = 0
        for x, _ in data:
            if(n_samples >= self.m_size):
                break
            # Projector representations
            z = F.normalize(self.target_projector(self.target_backbone(x[0].to(device))), dim=1)
            proj_bank.append(z)
            n_samples += x[0].shape[0]

        # Proj_bank: [m_size, D]
        proj_bank = torch.cat(proj_bank, dim=0).contiguous()
        if(n_samples > self.m_size):
            proj_bank = proj_bank[:self.m_size]

        # Save projections: size after saving [D, m_size]
        self.mem_projections.weight.data.copy_(proj_bank.T)

        # Reset BN parameters to original state
        self.load_state_dict(reset_dict, strict=False)


    @torch.no_grad()
    def update_memory(self, F):
        N = F.shape[0]
        # Shift memory [D, m_size]
        self.mem_projections.weight.data[:, :-N] = self.mem_projections.weight.data[:, N:].detach().clone()
        # Transpose LHS [D, bsize]
        self.mem_projections.weight.data[:, -N:] = F.T.detach().clone()


    # Target model's update
    @torch.no_grad()
    def update_target(self):
        tau = self.ema_value
        for target, online in zip(self.target_backbone.parameters(), self.backbone.parameters()):
            target.data = (tau) * target.data + (1 - tau) * online.data
        for target, online in zip(self.target_projector.parameters(), self.projector.parameters()):
            target.data = (tau) * target.data + (1 - tau) * online.data


    # Local clustering (happens at the client after every training round; clusters are made equally sized via Sinkhorn-Knopp, satisfying K-anonymity)
    def local_clustering(self, device='cuda:7'):
        # Local centroids: [# of centroids, D]; local clustering input (mem_projections.T): [m_size, D]
        with torch.no_grad():
            Z = self.mem_projections.weight.data.T.detach().clone()
            centroids = Z[np.random.choice(Z.shape[0], self.N_local, replace=False)]
            local_iters = 5
            # clustering
            for it in range(local_iters):
                assigns = sknopp(Z @ centroids.T, max_iters=10)
                choice_cluster = torch.argmax(assigns, dim=1)
                for index in range(self.N_local):
                    selected = torch.nonzero(choice_cluster == index).squeeze()
                    selected = torch.index_select(Z, 0, selected)
                    if selected.shape[0] == 0:
                        selected = Z[torch.randint(len(Z), (1,))]
                    centroids[index] = F.normalize(selected.mean(dim=0), dim=0)
                    
        # Save local centroids
        self.local_centroids.weight.data.copy_(centroids.to(device))


    # Global clustering (happens only on the server; see Genevay et al. for full details on the algorithm)
    def global_clustering(self, Z1, nG=1., nL=1.):
        N = Z1.shape[0] # Z has dimensions [m_size * n_clients, D]

        # Optimizer setup
        optimizer = torch.optim.SGD(self.centroids.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        train_loss = 0.
        total_rounds = 500

        for round_idx in range(total_rounds):
            with torch.no_grad():
                # Cluster assignments from Sinkhorn Knopp
                SK_assigns = sknopp(self.centroids(Z1))

            # Zero grad
            optimizer.zero_grad()

            # Predicted cluster assignments [N, N_centroids] = local centroids [N, D] x global centroids [D, N_centroids]
            probs1 = F.softmax(self.centroids(F.normalize(Z1, dim=1)) / self.T, dim=1)

            # Match predicted assignments with SK assignments
            loss = - F.cosine_similarity(SK_assigns, probs1, dim=-1).mean()

            # Train
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.centroids.weight.copy_(F.normalize(self.centroids.weight.data.clone(), dim=1)) # Normalize centroids
                train_loss += loss.item()

            progress_bar(round_idx, total_rounds, 'Loss: %.3f' % (train_loss/(round_idx+1))) 


    # Main training function
    def forward(self, x1, x2, x3=None, deg_labels=None):
        N = x1.shape[0]
        C = self.centroids.weight.data.detach().clone().T

        # Online model's outputs [bsize, D]
        Z1 = F.normalize(self.projector(self.backbone(x1)), dim=1)
        Z2 = F.normalize(self.projector(self.backbone(x2)), dim=1)

        # Compute online model's assignments 
        cZ2 = Z2 @ C

        # Convert to log-probabilities
        logpZ2 = torch.log(F.softmax(cZ2 / self.T, dim=1))

        # Target outputs [bsize, D]
        with torch.no_grad():
            self.update_target()
            tZ1 = F.normalize(self.target_projector(self.target_backbone(x1)), dim=1)

            # Compute target model's assignments
            cP1 = tZ1 @ C
            tP1 = F.softmax(cP1 / self.T, dim=1)

        # Clustering loss
        L_cluster = - torch.sum(tP1 * logpZ2, dim=1).mean()

        # Degeneracy regularization
        deg_preds = self.deg_layer(self.projector(self.backbone(x3)))
        L_deg = F.cross_entropy(deg_preds, deg_labels)
        L = L_cluster + L_deg

        # Update target memory
        with torch.no_grad():
            self.update_memory(tZ1) # New features are [bsize, D]
            
        return L

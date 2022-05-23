import torch
import torch.nn.functional as F
import torchvision
from models import create_backbone, simclr, simsiam, byol, specloss, rotpred, orchestra
import utils
from config import get_eval_dict, get_config_dict


# Alignment / Uniformity (adapted from Wang and Isola, ICML 2020)
def align_uniform(z1, z2, t=1, temperature=0.1):
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

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature
    labels = torch.zeros(2*N, device=device, dtype=torch.int64) # scalar label per sample
    loss = F.cross_entropy(logits, labels, reduction='sum')

    return (positives.mean()).item(), - (negatives.add(-1).mul(t).exp().sum() / (4 * N * (N-1))).log().item() / (2 * t), loss / (2 * N)


def main(config_dict, eval_dict):
    device = torch.device(eval_dict["main_device"] if torch.cuda.is_available() else "cpu")

    # Load model
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

    pretrained_model = torch.load(eval_dict['pretrained_loc'], map_location='cpu')
    net.load_state_dict(pretrained_model['net'], strict=False) 
    net = net.to(device)

    net.eval()

    n_iters = 0
    train_loss, train_align, train_uniform = 0, 0, 0

    with torch.no_grad():
        for client_id in range(config_dict['num_clients']):
            # Dataloaders
            trainloader, _, testloader = utils.load_data(config_dict, client_id=client_id, alpha=config_dict['alpha'], n_clients=config_dict['num_clients'], 
                bsize=eval_dict["batch_size"], hparam_eval=True, in_simulation=config_dict["virtualize"])

            for batch_idx, ((input1, input2), labels) in enumerate(trainloader):
                n_iters += 1
                input1, input2 = input1.to(device), input2.to(device)

                Z1, Z2 = F.normalize(net.projector(net.backbone(input1)), dim=1), F.normalize(net.projector(net.backbone(input2)), dim=1)
                align, unif, loss = align_uniform(Z1, Z2) 

                train_loss +=  loss.item() 
                train_align += align
                train_uniform += unif

            utils.progress_bar(client_id, config_dict['num_clients'], 'Loss: %.3f | Alignment: %.3f;  Uniformity: %.3f' 
                % (train_loss / (n_iters), train_align / (n_iters), train_uniform / (n_iters))) 

        # Server Dataloader (this is just for sanity check and not supposed to be used in practice)
        _, memloader, testloader = utils.load_data(config_dict, client_id=-1, bsize=256)
        knn_acc = utils.knn_monitor(net.backbone, memloader, testloader, device=device)


    return {
        'alignment': train_align / (n_iters),
        'uniformity': train_uniform / (n_iters),
        'obj': train_loss / (n_iters),
        'kNN': knn_acc,
    }

if __name__ == "__main__":
    print(main(get_config_dict(), get_eval_dict()))

import torch
import torch.nn as nn
import torchvision
from models import create_backbone
import utils
from config import get_eval_dict, get_config_dict

def main(config_dict, eval_dict):
    device = torch.device(eval_dict["main_device"] if torch.cuda.is_available() else "cpu")
    # Dataloaders
    trainloader, _, testloader = utils.load_data(config_dict, client_id=-1, bsize=eval_dict["batch_size"], linear_eval=True)

    # Model definitions
    net = create_backbone(name=eval_dict["model_class"], num_classes=0).to(device)
    classifier = nn.Linear(in_features=net.output_dim, out_features=len(trainloader.dataset.classes), bias=True).to(device)

    # Load model
    pretrained_model = torch.load(eval_dict["pretrained_loc"], map_location='cpu')
    net.load_state_dict({k[9:]:v for k, v in pretrained_model['net'].items() if k.startswith('backbone.')}, strict=True)    
    del pretrained_model
    net = net.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=0)

    # define lr scheduler
    lr_scheduler = utils.LR_Scheduler(optimizer, warmup_epochs=eval_dict["warmup_epochs"], warmup_lr=eval_dict["warmup_lr"], 
        num_epochs=eval_dict["num_epochs"], base_lr=eval_dict["base_lr"]*eval_dict["batch_size"]/256, 
        final_lr=eval_dict["final_lr"]*eval_dict["batch_size"]/256, iter_per_epoch=len(trainloader))

    # Train
    net.eval()
    classifier.train()
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(eval_dict["num_epochs"]):
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                features = net(inputs.to(device))
            outputs = classifier(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            lr = lr_scheduler.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f | LR: %.3f'
                % (train_loss/(batch_idx+1), 100.*correct/total, optimizer.param_groups[0]['lr']))

    # Test
    classifier.eval()
    correct, total, test_loss = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    print("\n")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = classifier(net(inputs.to(device)))
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return {
        'correct': correct,
        'total': total,
        'accuracy': correct/total,
        'test_loss': test_loss
    }

if __name__ == "__main__":
    print(main(get_config_dict(), get_eval_dict()))

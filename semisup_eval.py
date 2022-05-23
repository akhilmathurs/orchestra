import argparse
import distutils.util
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from models import create_backbone
import utils
from config import get_eval_dict, get_config_dict

torch.manual_seed(0)
cudnn.deterministic = True
cudnn.benchmark = False

def main(config_dict, eval_dict):
    device = torch.device(eval_dict["main_device"] if torch.cuda.is_available() else "cpu")
    # Dataloaders
    trainloader, _, testloader = utils.load_data(
        config_dict, client_id=-1, bsize=eval_dict["batch_size"], linear_eval=True,
        subset_proportion=eval_dict['subset_proportion'], subset_force_class_balanced=eval_dict['subset_force_class_balanced'], subset_seed=eval_dict['subset_seed']
    )

    # Model definitions
    net = create_backbone(name=eval_dict["model_class"], num_classes=0).to(device)
    classifier = nn.Linear(in_features=net.output_dim, out_features=len(trainloader.dataset.classes), bias=True).to(device)

    # Load model
    pretrained_model = torch.load(eval_dict["pretrained_loc"], map_location='cpu')
    if('sup' in eval_dict["pretrained_loc"]):
        net.load_state_dict({k:v for k, v in pretrained_model['net'].items() if not k.startswith('linear.')}, strict=True)    
    else:    
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
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(eval_dict["num_epochs"]):
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
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
    net.eval()
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

def get_parser():
    def strtobool(v):
        return bool(distutils.util.strtobool(v))
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--config_dict", type=str, default="{}")
    parser.add_argument("--eval_dict", type=str, default="{}")
    parser.add_argument("--pretrained_loc", type=str, default=None)
    parser.add_argument("--subset_proportion", type=float, default=None)
    parser.add_argument("--subset_force_class_balanced", type=strtobool, default=None)
    parser.add_argument("--subset_seed", type=int, default=None)
    return parser

if __name__ == "__main__":
    import parse_run
    import config
    import pprint
    from datetime import datetime

    run_time = datetime.now()
    time_string = run_time.strftime("%Y%m%d-%H%M%S%f")[:-3]

    parser = get_parser()
    args = parser.parse_args()
    config_dict = config.get_config_dict()
    eval_dict = config.get_eval_dict()
    config_dict, eval_dict = parse_run.update_configs(args, config_dict, eval_dict)
    if args.pretrained_loc is not None:
        eval_dict['pretrained_loc'] = args.pretrained_loc
    if args.subset_proportion is not None:
        eval_dict['subset_proportion'] = args.subset_proportion
    if args.subset_force_class_balanced is not None:
        eval_dict['subset_force_class_balanced'] = args.subset_force_class_balanced
    if args.subset_seed is not None:
        eval_dict['subset_seed'] = args.subset_seed

    eval_results = main(config_dict, eval_dict)

    pp = pprint.PrettyPrinter()
    with open(f'{config_dict["save_dir"]}/configs/eval_results_{time_string}.txt', 'w') as f:
        f.write(pp.pformat({
            'eval_results': eval_results,
            'eval_dict': eval_dict,
            'config_dict': config_dict
        }))
    print(eval_results)
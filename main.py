import argparse
import os
import pprint
import config
import server
import torch
import torch.backends.cudnn as cudnn
import linear_eval
cudnn.deterministic = True
cudnn.benchmark = False

def update_configs(args, config_dict, eval_dict):
    config_dict_update = eval(args.config_dict)
    config_dict.update(config_dict_update)

    eval_dict_update = eval(args.eval_dict)
    eval_dict.update(config_dict_update)
    eval_dict.update(eval_dict_update)
    if(config_dict["train_mode"]=="orchestra"):
        model_name = f'{config_dict["train_mode"]}_{config_dict["num_clients"]}_clients_{config_dict["local_bsize"]}_bsize_{config_dict["local_epochs"]}_lepochs_{config_dict["fraction_fit"]}_fit_{config_dict["num_global_clusters"]}_gclusters_{config_dict["num_local_clusters"]}_lclusters_{config_dict["seed"]}_seed'
    elif(config_dict["train_mode"]=="specloss"):
        model_name = f'{config_dict["train_mode"]}_{config_dict["num_clients"]}_clients_{config_dict["local_bsize"]}_bsize_{config_dict["local_epochs"]}_lepochs_{config_dict["fraction_fit"]}_fit_{config_dict["num_global_clusters"]}_specclusters_{config_dict["seed"]}_seed'
    else:
        model_name = f'{config_dict["train_mode"]}_{config_dict["num_clients"]}_clients_{config_dict["local_bsize"]}_bsize_{config_dict["local_epochs"]}_lepochs_{config_dict["fraction_fit"]}_fit_{config_dict["seed"]}_seed'
    eval_dict['pretrained_loc'] = f'{config_dict["save_dir"]}/saved_models/model_{config_dict["dataset"]}_{config_dict["alpha"]}_alpha_'+model_name+'.pth'
    return config_dict, eval_dict

def run(config_dict, eval_dict, do_linear=True):
    
    print("========================")
    print("Run Configurations:")
    print("config_dict:")
    print(config_dict)
    print("========================")

    os.makedirs(f'{config_dict["save_dir"]}/configs', exist_ok=True)
    pp = pprint.PrettyPrinter()
    with open(f'{config_dict["save_dir"]}/configs/config_dict.txt', 'w') as f:
        f.write(pp.pformat(config_dict))

    # Federated training
    server.server_run(config_dict)

    # Linear eval
    if(do_linear):
        eval_results = linear_eval.main(config_dict, eval_dict)

    # Save results
    if(config_dict["train_mode"]=="orchestra"):
        model_name = f'{config_dict["train_mode"]}_{config_dict["num_clients"]}_clients_{config_dict["local_bsize"]}_bsize_{config_dict["local_epochs"]}_lepochs_{config_dict["fraction_fit"]}_fit_{config_dict["num_global_clusters"]}_gclusters_{config_dict["num_local_clusters"]}_lclusters_{config_dict["seed"]}_seed'
    elif(config_dict["train_mode"]=="specloss"):
        model_name = f'{config_dict["train_mode"]}_{config_dict["num_clients"]}_clients_{config_dict["local_bsize"]}_bsize_{config_dict["local_epochs"]}_lepochs_{config_dict["fraction_fit"]}_fit_{config_dict["num_global_clusters"]}_specclusters_{config_dict["seed"]}_seed'
    else:
        model_name = f'{config_dict["train_mode"]}_{config_dict["num_clients"]}_clients_{config_dict["local_bsize"]}_bsize_{config_dict["local_epochs"]}_lepochs_{config_dict["fraction_fit"]}_fit_{config_dict["seed"]}_seed'
    eval_dest = f'{config_dict["save_dir"]}/saved_models/eval_{config_dict["dataset"]}_{config_dict["alpha"]}_alpha_'+model_name+'.txt'
    with open(eval_dest, 'w') as f:
        f.write(pp.pformat(eval_results))

    return eval_results

def get_parser():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--do_linear", type=bool, default=True)
    parser.add_argument("--config_dict", type=str, default="{}")
    parser.add_argument("--eval_dict", type=str, default="{}")
    return parser

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    config_dict = config.get_config_dict()
    eval_dict = config.get_eval_dict()
    config_dict, eval_dict = update_configs(args, config_dict, eval_dict)
    torch.manual_seed(config_dict['seed'])
    run(config_dict, eval_dict, do_linear=args.do_linear)
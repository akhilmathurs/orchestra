#!/bin/bash

# Training with linear eval
python main.py --config_dict="{'train_mode': 'orchestra', 'da_method': 'orchestra', 'local_lr': 0.003}" --do_linear=True

# Semi-supervised evaluation (assumes you have a pretrained model)
python semisup_eval.py --subset_proportion=0.1 --subset_force_class_balanced=True --pretrained_loc="./saved_models/model_CIFAR10_0.1_alpha_byol_100_clients_16_bsize_1_lepochs_0.5_fit_0_seed.pth"

# Tuning hyperparameters of a model (grid defined in hparam_main.py)
python hparam_main.py --config_dict="{'train_mode': 'orchestra', 'da_method': 'orchestra', 'num_rounds': 20, 'save_dir':'runs'}"

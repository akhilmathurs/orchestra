#!/bin/bash

for i in 0
do
	python main.py --config_dict="{'train_mode': 'simsiam', 'da_method': 'simsiam', 'seed': $i, 'local_lr': 0.01}"
done
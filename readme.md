# Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering

Codebase for the paper ["Orchestra: Unsupervised Federated Learning via Globally Consistent Clustering."](https://arxiv.org/abs/) 

Contains federated training with several unsupervised methods, including the proposed method [Orchestra](https://arxiv.org/abs/), [SimCLR](https://arxiv.org/abs/2002.05709), [SimSiam](https://arxiv.org/abs/2011.10566), [SpecLoss](https://arxiv.org/abs/2106.04156), [BYOL](https://arxiv.org/abs/2006.07733), and [Rotation Prediction](https://arxiv.org/abs/1803.07728); supports linear evaluation protocol; semi-supervised evaluation protocol; and hyperparameter tuning via the alignment-uniformity scheme discussed in the paper.

## Requirements

The code requires:

* Python 3.6 or higher

* Pytorch 1.9 or higher

We use the [Flower](https://flower.dev) federated learing framework for all client-server implementation. Flower and other dependencies can be installed via following command:

```setup
pip -r install requirements.txt
```

## Example execution 

First use the following command to setup the dataset of your choice (e.g., CIFAR-10) for any number of clients (e.g., 100) and heterogeneity (e.g., 0.1 as Dirichlet distribution parameter):

```sampler
python sampler.py --dataset="CIFAR10" --n_clients=100 --alpha=0.1
```

Then, to train a model using a particular training method and perform linear eval, run the following command

```execution
python main.py --config_dict="{'train_mode': 'orchestra', 'da_method': 'orchestra', 'local_lr': 0.003}" --linear_eval=True
```

Other execution examples (semi-supervised training and hyperparameter tuning) are provided in the ```examples.sh``` file.

## Organization

### -- Support files

* **myfedavg.py**: Manually defined Federated Averaging protocol to allow support server-level manipulation of the global model

* **config.py**: Contains a dict that defines all hyperparameters for federated training and GPU management.

* **sampler.py**: Splits the dataset into predefined number of clients.

* **utils.py**: Dataloaders, test functions, progress bars

### -- Federated training and Self-supervised learning 

* **server.py**: Server module; manages clients and records global progress.

* **client.py**: Client module; contains client-specific functions (e.g., unsupervised training protocol)

* **models.py**: All backbone and SSL function definitions (includes [Orchestra](https://arxiv.org/abs/), [SimCLR](https://arxiv.org/abs/2002.05709), [SimSiam](https://arxiv.org/abs/2011.10566), [SpecLoss](https://arxiv.org/abs/2106.04156), [BYOL](https://arxiv.org/abs/2006.07733)), [Rotation Prediction](https://arxiv.org/abs/1803.07728))

### -- Evaluation protocols 

* **linear_eval.py**: Linear evaluation protocol

* **semisup_eval.py**: Semi-supervised evaluation protocol

### -- Hyperparameter tuning 

* **hparam_method.py**: Implemented the unsupervised hyperparameter tuning protocol

* **hparam_parser.py**: Parses and evaluates one single point on the hyperparameter grid

* **hparam_main.py**: Main execution file for hyperparameter tuning that evaluates each grid point 


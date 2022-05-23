from typing import Callable, Dict, List, Optional, Tuple
from flwr.common import Parameters, Scalar, Weights, parameters_to_weights, weights_to_parameters, FitRes
from flwr.server.strategy.fedavg import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

import numpy as np
from collections import OrderedDict
import torch
import torch.nn.functional as F
from models import orchestra

class MyFedAvg(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        eta: float = 1e0,
        config_dict=None
    ) -> None:
        """FedAvg with a global learning rate
        The goal for making our own implementation was to allow global clustering and divergence aware updates.
        This may seem annoyingly weird, but the issue is communicating vectors was non-trivial in Flower 
        and this hack was a straight-forward answer.
        Args:
            fraction_fit (float, optional): Fraction of clients used during
                training. Defaults to 0.1.
            fraction_eval (float, optional): Fraction of clients used during
                validation. Defaults to 0.1.
            min_fit_clients (int, optional): Minimum number of clients used
                during training. Defaults to 2.
            min_eval_clients (int, optional): Minimum number of clients used
                during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of total
                clients in the system. Defaults to 2.
            eval_fn (Callable[[Weights], Optional[Tuple[float, float]]], optional):
                Function used for validation. Defaults to None.
            on_fit_config_fn (Callable[[int], Dict[str, str]], optional):
                Function used to configure training. Defaults to None.
            on_evaluate_config_fn (Callable[[int], Dict[str, str]], optional):
                Function used to configure validation. Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds
                containing failures. Defaults to True.
            initial_parameters (Parameters): Initial set of parameters from the server.
            eta (float, optional): Server-side learning rate. Defaults to 1e-1.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
        )
        self.current_weights = parameters_to_weights(initial_parameters)
        self.eta = eta
        self.config_dict = config_dict

    def __repr__(self) -> str:
        rep = f"FedOpt(accept_failures={self.accept_failures})"
        return rep


    def __repr__(self) -> str:
        rep = f"FedAdam(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        config_dict = self.config_dict
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]
        num_successful_clients = len(weights_results) # No. of successful clients

        fedavg_parameters_aggregated, metrics_aggregated = weights_to_parameters(aggregate(weights_results)), {}

        if fedavg_parameters_aggregated is None:
            return None, {}

        # Aggregate weights, as you would for FedAvg
        fedavg_weights_aggregate = parameters_to_weights(fedavg_parameters_aggregated)

        # Update number of rounds
        if(fedavg_weights_aggregate[0].shape[0]==1):
            fedavg_weights_aggregate[0] = np.array(int(fedavg_weights_aggregate[0]) + 1.) 

        # Compute difference w.r.t. current parameters 
        delta = [x - y for x, y in zip(fedavg_weights_aggregate, self.current_weights)]

        # Divergence aware update for BYOL and SimSiam (used when in cross-silo settings to follow Zhuang et al.'s implementation)
        if((config_dict['train_mode']=='simsiam' or config_dict['train_mode']=='byol') and config_dict['div_aware_update']):
            div_sum = 0
            for l_data in delta: 
                div_sum += np.linalg.norm(l_data)
            if(div_sum > 0.5): # The threshold is a hyperparameter; -8 to -1 are predictor layers
                for l_num in range(-8, 0):
                    delta[l_num] = np.zeros_like(delta[l_num])

        # Move in the direction of the difference with a global lr
        new_weights = [x + self.eta * d for x, d in zip(self.current_weights, delta)]

        # Set weights
        self.current_weights = new_weights


        ######## Orchestra's server ops (i.e., global clustering) ######## 
        # Collect representations
        if(config_dict['train_mode']=='orchestra'):
            print("\nRetrieving Representations... ")
            device = torch.device(config_dict['main_device'] if torch.cuda.is_available() else "cpu")

            # Local clusters from clients [N_local * no. of clients, D]
            Z1 = np.array(weights_results[0][0][-1]) # -1 is local centroids 
            for client_data in weights_results[1:]:
                Z1 = np.concatenate((Z1, client_data[0][-1]), axis=0) # -1 is local centroids

            # Convert to tensor and transpose (local centroids need transpose)
            Z1 = torch.tensor(Z1, device=device).T

            ### Global clustering (Input should be of size [# of samples, D])
            net = orchestra(config_dict=config_dict, bbone_arch=config_dict['model_class']).to(device)

            # Load parameters
            params_dict = zip(net.state_dict().keys(), self.current_weights)
            state_dict = OrderedDict({k: torch.Tensor(np.array([v])) if (v.shape == ()) else torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=False)

            # Run global clustering
            net.global_clustering(Z1.to(device).T, nG=config_dict['num_global_clusters'], nL=config_dict['cluster_m_size'] * num_successful_clients) 

            # Retrieve trained parameters and update fedavg output
            self.current_weights = [val.cpu().numpy() for _, val in net.state_dict().items()]

            # Delete network to free memory; it's not needed anymore
            del net

        return weights_to_parameters(self.current_weights), metrics_aggregated

from data_gen import generate_data
import matplotlib.pyplot as plt
from tqdm import tqdm
import arviz as az
import numpy as np
import pymc as pm
import pytensor
import seaborn as sns
import sys
import os
import logging


# file some configs 
logger = logging.getLogger('pymc')
logger.setLevel(logging.WARNING)
floatX = pytensor.config.floatX
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")


def get_data(config):
    data_dict = generate_data(**config)
    X, y = data_dict['features'], data_dict['labels']
    X, y = np.asarray(X), np.asarray(y)
    return X, y

def construct_nn(ann_input, ann_output, n_hidden=32):
    X_train, y_train = ann_input, ann_output
    # Initialize random weights between each layer
    init_1 = rng.standard_normal(size=(X_train.shape[1], n_hidden)).astype(floatX)
    initb_1 = rng.standard_normal(size=n_hidden).astype(floatX)
    init_2 = rng.standard_normal(size=(n_hidden, n_hidden)).astype(floatX)
    initb_2 = rng.standard_normal(size=n_hidden).astype(floatX)
    init_out = rng.standard_normal(size=n_hidden).astype(floatX)

    coords = {
        "hidden_layer_1": np.arange(n_hidden),
        "hidden_layer_2": np.arange(n_hidden),
        "train_cols": np.arange(X_train.shape[1]),
        "obs_id": np.arange(X_train.shape[0]),
    }
    with pm.Model(coords_mutable=coords) as neural_network:
        ann_input = pm.Data("ann_input", X_train, dims=("obs_id", "train_cols"), mutable=True)
        ann_output = pm.Data("ann_output", y_train, dims="obs_id", mutable=True)

        # Weights from input to hidden layer
        weights_in_1 = pm.Normal(
            "w_in_1", 0, sigma=1, initval=init_1, dims=("train_cols", "hidden_layer_1")
        )
        
        biases_in_1 = pm.Normal(
            "b_in_1", 0, sigma=1, initval=initb_1, dims="hidden_layer_1"
        )

        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal(
            "w_1_2", 0, sigma=1, initval=init_2, dims=("hidden_layer_1", "hidden_layer_2")
        )

        biases_1_2 = pm.Normal(
            "b_1_2", 0, sigma=1, initval=initb_2, dims="hidden_layer_2"
        )
        # Weights from hidden layer to output
        weights_2_out = pm.Normal("w_2_out", 0, sigma=1, initval=init_out, dims="hidden_layer_2")

        # Build neural-network using tanh activation function
        act_1 = pm.math.sigmoid(pm.math.dot(ann_input, weights_in_1) + biases_in_1)
        act_2 = pm.math.sigmoid(pm.math.dot(act_1, weights_1_2) + biases_1_2)
        act_out = pm.math.sigmoid(pm.math.dot(act_2, weights_2_out))

        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli(
            "out",
            act_out,
            observed=ann_output,
            total_size=y_train.shape[0],  # IMPORTANT for minibatches
            dims="obs_id",
        )
    return neural_network

# In-distribution and out-of-distribution configurations
id_config = {
    'n': 1000,
    'f': lambda x : sum([np.sin(c) for c in x]),
    'mean': np.pi,
    'var': 1,
    'd': 10,
    'gap': 0,
}
ood_config = {
    'n': 1000,
    'f': lambda x : sum([np.sin(c) for c in x]),
    'mean': np.pi+1,
    'var': 1,
    'd': 10,
    'gap': 0,
}

def pretraining(X_train, 
                y_train, 
                id_config,
                n_iterations=30000, 
                plot=True, 
                base_trace_draws=5000,
                Phi_size=500,
                dis_rate_sample_size=500):
    neural_network = construct_nn(X_train, y_train)
    
    # Train Bayesian neural network
    with neural_network:
        print('Fitting Bayesian neural network to id training data...')
        approx = pm.fit(n=n_iterations, method='advi')  
        
    if plot:  
        plt.plot(approx.hist, alpha=0.3)
        plt.ylabel("ELBO")
        plt.xlabel("Iteration");
        plt.savefig(os.path.join('plots', 'ELBO.png'))
    
    # Draw from the posterior p(w|X_train, y_train) for pseudolabels
    base_trace = approx.sample(draws=base_trace_draws)
    
    # Building disagreement distribution Phi
    Phi = []
    for i in tqdm(range(Phi_size)):
        # get new dataset
        X_, y_ = get_data(id_config)
        X_, y_ = X_.astype(floatX), y_.astype(floatX)
        
        # get trace for this round 
        trace = approx.sample(draws=dis_rate_sample_size)
        
        with neural_network:
            pm.set_data(new_data={"ann_input": X_})
            ppc = pm.sample_posterior_predictive(base_trace, progressbar=False)
            
            # get base model to label X_
            y_ = np.expand_dims(ppc.posterior_predictive["out"].mean(("chain", "draw")) > 0.5, axis=0) # label
            ppc = pm.sample_posterior_predictive(trace, progressbar=False)
            y_hats = np.squeeze(ppc.posterior_predictive["out"]) > 0.5 # (num_draws, n_samples)
            
            # compare
            disagreement_matrix = (np.tile(y_, (dis_rate_sample_size, 1)) != y_hats)
            disagreement_rates = np.sum(disagreement_matrix, axis=1)/disagreement_matrix.shape[1]
            Phi.append(np.max(disagreement_rates))
    
    return base_trace, neural_network, Phi, approx

def dpddm_test(base_trace, neural_network, Phi, approx, config, 
               runs=10, 
               dis_rate_sample_size=500):
    tprs = []
    dis_rates = []
    for i in tqdm(range(runs)):
        X_, y_ = get_data(config)
        X_, y_ = X_.astype(floatX), y_.astype(floatX)

        # get trace for this round 
        trace = approx.sample(draws=dis_rate_sample_size)

        with neural_network:
            pm.set_data(new_data={"ann_input": X_})
            ppc = pm.sample_posterior_predictive(base_trace, progressbar=False)
            # get base model to label X_
            y_ = np.expand_dims(ppc.posterior_predictive["out"].mean(("chain", "draw")) > 0.5, axis=0) # label
            
            pm.set_data(new_data={"ann_input": X_})
            ppc = pm.sample_posterior_predictive(trace, progressbar=False)
            y_hats = np.squeeze(ppc.posterior_predictive["out"]) > 0.5 # (num_draws, n_samples)
            
            # compare
            #idx = np.random.choice(np.arange(1000), 50)
            disagreement_matrix = (np.tile(y_, (dis_rate_sample_size, 1)) != y_hats)#[:, :]
            disagreement_rates = np.sum(disagreement_matrix, axis=1)/disagreement_matrix.shape[1]
            max_dis_rate = np.max(disagreement_rates)
            dis_rates.append(max_dis_rate)
        tmp = (max_dis_rate > np.quantile(Phi, 0.95)).values
        tprs.append(tmp)
    return tprs, dis_rates
        
def main():
    # Make in-distribution training dataset
    X_train, y_train = get_data(id_config)
    X_train = X_train.astype(floatX)
    y_train = y_train.astype(floatX)

    base_trace, neural_network, Phi, approx = pretraining(X_train, y_train, id_config)
    #plt.hist(Phi, bins=20)
    #plt.savefig(os.path.join('plots', 'Phi_distribution.png'))
    print('Phi 95th quantile: {:.4f}'.format(np.quantile(Phi, 0.95)))
    test_runs = 10
    # in-distribution test
    print('Running in-distribution D-PDDM test {} times'.format(test_runs))
    id_flag_list, id_dis_rates = dpddm_test(base_trace, neural_network, Phi, approx, id_config, runs=test_runs)
    print('In-distribution detection rate: {:.3f}'.format(sum(id_flag_list)/len(id_flag_list)))
    print('In-distribution disagreement rates:')
    print(id_dis_rates)
    # out-of-distribution test
    print('Running out-of-distribution D-PDDM test {} times'.format(test_runs))
    ood_flag_list, ood_dis_rates = dpddm_test(base_trace, neural_network, Phi, approx, ood_config, runs=test_runs)
    print('Out-of-distribution detection rate: {:.3f}'.format(sum(ood_flag_list)/len(ood_flag_list)))
    print('Out-of-distribution disagreement rates:')
    print(ood_dis_rates)

    
if __name__ == '__main__':
    main()
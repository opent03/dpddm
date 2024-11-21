import numpy as np


def generate_data(n, f, mean, var, var2=1, d=2,  gap=0, flip=False):
    """Generates two-dimensional dataset based on labeling function

    Args:
        n (int): dataset size
        f (float -> float): function 
        mean (float): x-axis mean of data
        var (float): x-axis variance of data
        d (int): dimension of the space of features. Defaults to 2, needs to be greater than 2
        gap (float, optional): gap between function and data generated. Defaults to 0
        flip (bool, optional): whether correct labels are flipped_. Defaults to False.
        
    Returns:
        dict containing features and 0-1 labels 
    """
    
    # Generate independent component of data
    # Assume isotropic Gaussian
    means = mean * np.ones(shape=(n, d-1))
    vrs = var * np.ones(shape=(n, d-1))
    x1 = np.random.normal(means, vrs, size=(n, d-1))
    eps = np.random.normal(0, var2, size=n)
    labels = np.sign(eps)
    if flip:
        labels = labels * (-1)
        
    #convert to 0,1
    labels = [int(p) if p==1 else 0 for p in labels]
    
    # Generate dependent component of data
    x2 = np.array(list(map(f, x1))) + eps + np.sign(eps)*gap
    x2 = np.expand_dims(x2, axis=1)
    # Merge x1 and x2
    features = np.concatenate([x1, x2], axis=1)
    return {'features': features, 'labels': labels}
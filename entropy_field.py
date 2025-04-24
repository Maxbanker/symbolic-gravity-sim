"""
entropy_field.py
Generates entropy exposure fields for Ψeff simulations
"""

import numpy as np

def generate_entropy_field(shape=(100, 100), method="gaussian", seed=None, **kwargs):
    """
    Create an entropy exposure field S(x, t)

    Args:
        shape (tuple): Size of the field grid
        method (str): Type of entropy field ('gaussian', 'uniform', 'gradient')
        seed (int, optional): Random seed for reproducibility
        kwargs: Parameters for field generation

    Returns:
        entropy_field (np.ndarray): Generated entropy field
    """
    if seed is not None:
        np.random.seed(seed)

    if method == "gaussian":
        mean = kwargs.get("mean", 0.5)
        std = kwargs.get("std", 0.1)
        field = np.clip(np.random.normal(loc=mean, scale=std, size=shape), 0, 1)

    elif method == "uniform":
        low = kwargs.get("low", 0.0)
        high = kwargs.get("high", 1.0)
        field = np.random.uniform(low, high, size=shape)

    elif method == "gradient":
        direction = kwargs.get("direction", "x")
        if direction == "x":
            field = np.tile(np.linspace(0, 1, shape[1]), (shape[0], 1))
        else:
            field = np.tile(np.linspace(0, 1, shape[0]), (shape[1], 1)).T

    else:
        raise ValueError("Unsupported entropy field method: {}".format(method))

    return field

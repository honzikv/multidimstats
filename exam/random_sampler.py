from typing import Callable

import numpy as np


def rejection_sampling(func: Callable[[np.array], np.array],
                       fx_max: float,
                       bounds_start: np.array,
                       bounds_end,
                       boundary_conditions: Callable[[np.array], np.array],
                       n_samples: int,
                       random_generator,
                       return_rejected=False,
                       sampling_batch_size=256):
    """
    Generate samples from a function using rejection sampling
    :param func: function to sample, function takes a batch of samples as input and returns a batch of function values
    :param fx_max: maximum value of the function in the bounds - scalar
    :param bounds_start: lower bounds of the function - 2D array with shape (1, n)
    :param bounds_end: upper bounds of the function - 2D array with shape (1, n)
    :param boundary_conditions: function that takes a vector of samples as input and returns a vector
                                of boolean values of shape (n_samples, 1)
    :param n_samples: number of samples to generate
    :param random_generator: random generator
    :param return_rejected: if True, return the rejected samples as well. If enabled ignores n_samples and
                            returns all the sampled elements
    :param sampling_batch_size: number of samples to generate in each batch
    :return: array of samples with shape (n_samples, n)
    """
    x_dim = bounds_start.shape[1]  # Dimensionality of the vector x
    batch_size = sampling_batch_size  # Number of samples to generate in each batch

    def sample_batch():
        # Generate vectors u, v with uniform random numbers in (0, 1)
        # I.e. we get a vector form of u_i ~ U(0, 1) for each dim i
        u = random_generator.uniform(size=(batch_size, x_dim))

        # Generate a vector of random numbers in (0, 1) for each sample
        # This represents the function value scale factor at the sampled point
        v = random_generator.uniform(0, 1, batch_size)

        # Scale the vectors u, v to the bounds of the function
        # in a vector fashion
        x = bounds_start + (bounds_end - bounds_start) * u

        # Calculate the function value at the sampled points
        return x, func(x), v

    # Create array with all the accepted samples
    accepted, rejected = [], []

    # While we haven't collected enough samples
    samples_generated = 0
    while samples_generated < n_samples:
        # Generate a batch of samples
        x, fx, v = sample_batch()

        # Filter out invalid samples
        accepted_filter = ((v * fx_max) < fx) & boundary_conditions(x)
        valid_samples = x[accepted_filter]

        # Add the valid samples to the array
        samples_generated += len(valid_samples)
        accepted.append(valid_samples)

        # Add the invalid samples to the array
        if return_rejected:
            rejected.append(x[~accepted_filter])

    # Concatenate all the batches into a single array
    if return_rejected:
        return accepted, rejected

    return np.concatenate(accepted)[:n_samples]


# %%

# Define the function to sample and its conditions
f = lambda x: 1 / 14 * x[:, 0] * (x[:, 0] + x[:, 1])  # f(x, y) = 1/14 * x (x + y)
f_conditions = lambda x: x[:, 0] < x[:, 1]  # x < y

# Define the bounds of the function
# Bounds must be a 2D array with shape (1, n)
bounds_start = np.array([[1, 1]])  # x >= 1 and y >= 1
bounds_end = np.array([[3, 3]])  # x <= 2 and y <= 2

# Calculate the maximum value of the function in the bounds
fx_max = f(bounds_end).item()  # Maximum value of the function in the bounds

# Number of samples to generate
n_samples = 1_000_000

# Random generator
random_generator = np.random.default_rng(seed=420)

# Generate the samples
samples = rejection_sampling(
    func=f,
    fx_max=fx_max,
    bounds_start=bounds_start,
    bounds_end=bounds_end,
    boundary_conditions=f_conditions,
    n_samples=n_samples,
    random_generator=random_generator,
)

# Compute mean for x1
print(f'Mean for x1: {np.mean(samples[:, 0])}')
print(f'Mean for x2: {np.mean(samples[:, 1])}')
print(f'Variance for x1: {np.var(samples[:, 0])}')
print(f'Variance for x2: {np.var(samples[:, 1])}')
print(f'Covariance: {np.cov(samples[:, 0], samples[:, 1])}')

# %%

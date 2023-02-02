import math
import numpy as np


def nd_hyper_geometric(x: np.array, M: np.array):
    """
    Calculate the probability of selecting x elements from each class in M
    :param x: The number of elements to select from each class
    :param M: The number of elements in each class
    :return: The probability of selecting x elements from each class in M
    """
    total_elements = np.sum(M)  # M is a vector of counts for each class
    total_elements_selection = np.sum(x)  # x is the number of elements to select from each class

    # Calculate the number of ways to select x elements from each class
    numerator = np.prod([math.comb(M[i], x[i]) for i in range(len(M))])
    denumerator = math.comb(total_elements, total_elements_selection)

    return numerator / denumerator


#%%

# Test the function

classes = ['red', 'green', 'blue']
class_counts = [10, 20, 30]  # 10 red balls, 20 green balls, 30 blue balls

# What is the probability that given 10 draws we get 5 red, 3 green and 2 blue balls?
x = [5, 3, 2]

print(nd_hyper_geometric(x, class_counts) * 100)  # 16.7%


#%%



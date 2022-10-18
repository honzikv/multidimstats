import math


def mean(arr, n):
     
    summ = 0
    for i in range(n):
        summ = summ + arr[i]    
    return summ / n

# Function to calculate standard
# deviation of data.
def standardDeviation(arr,n):
     
    summ = 0
     
    # find standard deviation
    # deviation of data.
    for i in range(n):
        summ = (arr[i] - mean(arr, n)) *(arr[i] - mean(arr, n))
     
    return math.sqrt(summ / n)

# Function to calculate skewness.
def skewness(arr, n):
     
    # Find skewness using above formula
    summ = 0
    for i in range(n):
        summ = (arr[i] - mean(arr, n))*(arr[i] - mean(arr, n))*(arr[i] - mean(arr, n))
    return summ / (n * standardDeviation(arr, n) *standardDeviation(arr, n) *standardDeviation(arr, n) * standardDeviation(arr, n))
 
# Driver function
 
arr = [2.5, 3.7, 6.6, 9.1,9.5, 10.7, 11.9, 21.5,22.6, 25.2]
                 
# calculate size of array.
n = len(arr)
 
# skewness Function call
print('%.6f'%skewness(arr, n))

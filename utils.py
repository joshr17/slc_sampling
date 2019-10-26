import numpy as np
import scipy
from random import random
from bisect import bisect
from scipy.stats import ortho_group


def weighted_random(P):
    cdf = [P[0]]
    for i in range(1, len(P)):
        cdf.append(cdf[-1] + P[i])
    
    random_ind = bisect(cdf,random())
    
    return random_ind
        
def rand_PSD(size):
    A = np.random.rand(size,size)
    B = np.dot(A,A.transpose())
    return B  
    

def choose_evals_PSD(evals):
    size = evals.shape[0]
    diagonal = np.diag(evals)
    O = ortho_group.rvs(size)
    PSD = np.dot( np.transpose(O), diagonal)
    PSD = np.dot(PSD, O)
    return PSD
    
    
def set_to_binary(set_list, ground_size):
    if type(set_list) == list:
        binary_list = []
        for st in set_list:
            st = list(map(int, st))
    
            binary = np.zeros(ground_size)
            binary[st] = 1
            binary_list.append(binary )
            
        return binary_list

    if type(set_list) == np.ndarray:
        binary_array = np.zeros((set_list.shape[0], ground_size))        
        for index, st in enumerate(set_list):
            st = np.array(list(map(int, st)))
            binary = np.zeros(ground_size)
            binary[st] = 1
            binary_array[index] = binary
        return binary_array

def factorial(a,b):
    num = 1
    for i in range(a+1,b+1):
        num = num * i
    return num


def binary_to_set(binary_list):
    set_list = []
    for binary in binary_list:
        set_list.append(np.nonzero(binary))
    return set_list

def list_to_array(list_of_arrays):
    width = list_of_arrays[0].shape[0]
    height = len(list_of_arrays)
    
    arr = np.zeros((height,width))
    
    for i in range(height):
        arr[i] = list_of_arrays[i]
    return arr



def get_eig(L, flag_gpu=False):
    if flag_gpu:
        pass
    else:
        return scipy.linalg.eigh(L)

def get_sympoly(D, k, flag_gpu=False):
    N = D.shape[0]
    if flag_gpu:
        pass
    else:
        E = np.zeros((k+1, N+1))

    E[0] = 1.
    for l in range(1,k+1):
        E[l,1:] = np.copy(np.multiply(D, E[l-1,:N]))
        E[l] = np.cumsum(E[l], axis=0)

    return E


def gershgorin(A):
    radius = np.sum(np.absolute(A), axis=0)
    
    lambda_max = np.max(radius)
    lambda_min = np.min(2 * np.diag(A) - radius)

    return lambda_min, lambda_max


def kpp(X, k, flag_kernel=False):
    # if X is not kernel, rows of X are samples

    N = X.shape[0]
    rst = np.zeros(k, dtype=int)
    rst[0] = np.random.randint(N)

    if flag_kernel:
        # kernel kmeans++
        v = np.ones(N) * np.inf
        for i in range(1, k):
            Y = np.diag(X) + np.ones(N)*X[rst[i-1],rst[i-1]] - 2*X[rst[i-1]]
            v = np.minimum(v,Y)
            r = np.random.uniform()
            rst[i] = np.where(v.cumsum() / v.sum() >= r)[0][0]

    else:
        # normal kmeans++
        centers = [X[rst[0]]]
        for i in range(1, k):
            dist = np.array([min([np.linalg.norm(x-c)**2 for c in centers]) for x in X])
            r = np.random.uniform()
            ind = np.where(dist.cumsum() / dist.sum() >= r)[0][0]
            rst[i] = ind
            centers.append(X[ind])

    return rst

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll


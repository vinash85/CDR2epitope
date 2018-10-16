#  this module are  python equivalent of R functions
import numpy as np


def intersect(lst1, lst2):
    return list(set(lst1) & set(lst2))


def unique(aa):
    return list(set(aa))


# def sample(aa):
    # change later to take multiple parameters
    # return np.random.shuffle(aa)


def which(lst):
    return list(np.where(lst)[0])


def rowSums(aa):
    return np.sum(aa, axis=1)


def colSums(aa):
    return np.sum(aa, axis=0)


def rowMeans(aa):
    return np.mean(aa, axis=1)


def colMeans(aa):
    return np.mean(aa, axis=0)


def rbind(aa, bb):
    return np.concatenate((aa, bb), axis=0)


def cbind(aa, bb):
    return np.concatenate((aa, bb), axis=1)


def cbind_array_mat(array, mat):
    return cbind(np.tile(array, (len(mat), 1)), mat)


def sample(List, size=1, replace=False, prob=None):
    '''
    sample.int(n=n, size = s, replace = FALSE, prob = NULL)
    '''
    if prob is not None:
        prob = prob / sum(prob)
    return np.random.choice(List, size=size, replace=replace, p=prob)


def sample_int(n, size=1, replace=False, prob=None):
    '''
    sample.int(n=n, size = s, replace = FALSE, prob = NULL)
    prob should be np.array
    '''
    List = range(n)
    if prob is not None:
        prob = prob / sum(prob)

    return np.random.choice(List, size=size, replace=replace, p=prob)


def np_which_max(a):
    return np.argmax(a)


def which_max(a):
    return a.index(max(a))


def which_min(a):
    return a.index(min(a))


def setdiff(a, b):
    return list(set(a) - set(b))

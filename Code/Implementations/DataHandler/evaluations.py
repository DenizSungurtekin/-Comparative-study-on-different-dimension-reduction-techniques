import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import spearmanr

def stress(data1,data_red):
    distances1 = euclidean_distances(data1, data1)
    distances2 = euclidean_distances(data_red, data_red)
    num = np.sum(np.power(distances1-distances2,2))
    normalization = np.sum(np.power(distances1,2))
    stress = np.sqrt(num/normalization)

    return stress



def computeRank(distances1,distances2):
    rank1 = []
    rank2 = []

    for el,el2 in zip(distances1,distances2):  # Compute rank matrix for both distance (ascending order)
        rank1.append(sorted(range(len(el)), key=lambda k: el[k]))
        rank2.append(sorted(range(len(el2)), key=lambda k: el2[k]))

    rank1 = np.asarray(rank1)
    rank2 = np.asarray(rank2)

    return rank1,rank2

def rank_row(row1,row2):
    T = len(row1)
    diff = np.power(row1-row2,2)
    num = 6 * np.sum(diff)
    res = num/(np.power(T,3)-T)

    return 1-res


def spearman(data1,data_red):
    distances1 = euclidean_distances(data1, data1) # Compute distance matrix
    distances2 = euclidean_distances(data_red, data_red)
    rhos = []

    rank1,rank2 = computeRank(distances1,distances2)

    rank1 += 1
    rank2 += 1
    rank1 = np.asarray(rank1)
    rank2 = np.asarray(rank2)

    for row1,row2 in zip(rank1,rank2):
        rhos.append(rank_row(row1,row2))

    rhos = np.asarray(rhos)

    return np.mean(rhos)

def konig(data1,data_red,k1,k2):
    if k1 >= k2:
        print("K1 need to be smaller than k2")
        return "K1 need to be smaller than k2"

    distances1 = euclidean_distances(data1, data1) # Compute distance matrix
    distances2 = euclidean_distances(data_red, data_red)
    n = len(distances1)
    rank1,rank2 = computeRank(distances1,distances2)
    kms = []
    m = len(rank1)

    for i in range(m):
        km = []
        for j in range(m):
            if rank1[i][j] == rank2[i][j]:
                km.append(3)
                continue

            start = j-k1 # To search neighborhood at left but if j<k1 then we start at index 0
            if start < 0:
                start = 0

            if rank1[i][j] in rank2[i][start:j+k1+1]:
                km.append(2)
                continue
            start = j-k2
            if rank1[i][j] in rank2[i][start:j + k2 + 1]:
                km.append(1)
                continue
            else:
                km.append(0)

        kms.append(km)

    kms = np.asarray(kms)
    # print(kms[0,:]) To display result for a row
    # print(rank1[0,:])
    # print(rank2[0, :])

    # print(np.sum(kms[:,0:k1]))

    num = np.sum(kms[:,0:k1])
    score = num/(3*k1*n)
    return score


def meanRelativeError(data1,data_red,K):
    distances1 = euclidean_distances(data1, data1) # Compute distance matrix
    distances2 = euclidean_distances(data_red, data_red)
    n = len(distances1)

    rank1,rank2 = computeRank(distances1,distances2)
    rank1 += 1 # we want to start at rank 1
    rank2 += 1

    C = n*sum([np.abs(2*k-n-1)/k for k in range(1,K+1)])

    diff = np.abs(rank1 - rank2)

    mrre_X = diff/rank1
    mrre_Y = diff/rank2


    mrre_X = mrre_X[:,0:K+1] # We only take K neighbours to compute the final score, as the first rank is 0 for the point itself I add 1 to K
    mrre_Y = mrre_Y[:,0:K+1]

    score_X = np.sum(mrre_X)/C
    score_Y = np.sum(mrre_Y)/C

    return score_X,score_Y
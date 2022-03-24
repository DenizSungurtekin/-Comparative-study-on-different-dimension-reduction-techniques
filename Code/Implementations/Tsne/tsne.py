from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

def compute_proba_p_row(distance,sigma,index):
    probas = np.exp(-(distance)/(2*(np.power(sigma,2))))
    probas[index] = 1e-6  # not zero to avoid instability when computing entropy
    return probas/probas.sum()

def compute_perplexity(proba_row):
    entropy = -np.sum(proba_row*np.log2(proba_row))
    return np.power(2,entropy)

def find_optimal_sigma(target_perplexity,distance_row,row_index,low = 1,upper = 100,tol = 10e-5,max_ite = 50): #Binary search
    for i in range(max_ite):
        sigma = (low + upper) / 2.
        perplexity_current = compute_perplexity(compute_proba_p_row(distance_row, sigma, row_index))
        if np.abs(perplexity_current-target_perplexity) < tol:
            break
        else:
            if perplexity_current < target_perplexity:
                low = sigma
            else:
                upper = sigma
    return sigma

def compute_prob_p(distance,sigmas):
    N = distance.shape[0]
    matrix = np.concatenate([[compute_proba_p_row(distance[i],sigmas[i],i) for i in range(N)]],axis = 0)
    return matrix

def negatif_squared_distance(X):
    # Compute matrix of negative squared euclidean
    # c.f https://stackoverflow.com/questions/37009647
    somme = np.sum(np.square(X), 1)
    distance_sqr = np.add(np.add(-2 * np.dot(X, X.T), somme).T, somme)
    return -distance_sqr

def compute_q(Y):
    q = np.power(1. - negatif_squared_distance(Y), -1)
    np.fill_diagonal(q, 0.)  # qi,i
    return q / np.sum(q), q  # normalize in the entire matrix and return also the inverse distance

def p_conditional_to_joint(P):
    return (P + P.T) / (2. * P.shape[0])

def compute_joint_proba(data, target_perplexity):  # Compute joint probability from data and target perplexity
    N = data.shape[0]
    distances = np.square(euclidean_distances(data,data))
    sigmas = np.asarray([find_optimal_sigma(target_perplexity,distances[i],i) for i in range(N)])
    p_conditional = compute_prob_p(distances, sigmas)
    P = p_conditional_to_joint(p_conditional)
    return P

def tsne_grad(P, Q, Y, inv_distances):
    pq_sub_exp = np.expand_dims(P - Q, 2)  # (pij −qij)
    y_sub = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)  # (yi −y)
    dist_exp = np.expand_dims(inv_distances, 2)  # (1+||yi−yj||2)**-1
    y_diffs_wt = y_sub * dist_exp
    # We sum over the j c.f formula gradient
    grad = 4. * (pq_sub_exp * y_diffs_wt).sum(1)  # (pij−qij)(yi−yj)(1+||yi−yj||2)−1
    return grad


def tsne_reduction(data, target_dim, target_perplexity, iterations = 200, lr=0.01, momentum=False):
    # There is a lot of different initialisation method: The usual initialization routine for t-SNE is to start from a small random gaussian distribution, TODO
    # and use “early exaggeration” for the first 100 or so iterations where the input probabilities are multiplied by 4
    Y = np.random.normal(0, 1, size=(data.shape[0], target_dim)) * 4
    P = compute_joint_proba(data, target_perplexity)

    if momentum:
        Y_m2 = Y.copy()
        Y_m1 = Y.copy()

    for i in range(iterations):

        Q, inverse_distances = compute_q(Y)
        grads = tsne_grad(P, Q, Y, inverse_distances)

        Y = Y - lr * grads
        if momentum:
            Y += momentum * (Y_m1 - Y_m2)
            Y_m2 = Y_m1.copy()
            Y_m1 = Y.copy()

    return Y

N = 10
M = 5
PERPLEXITY = 5
TARGET_DIMENSION = 2
data = np.random.normal(0, 1, size=(N, M)) * 4
print(tsne_reduction(data,TARGET_DIMENSION,PERPLEXITY))
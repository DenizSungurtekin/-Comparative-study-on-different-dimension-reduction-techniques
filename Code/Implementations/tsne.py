from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
import warnings
from scipy.special import rel_entr
warnings.filterwarnings("ignore")

## Personal implementation of t-sne

def compute_proba_p_row(distance,sigma,index):  # For each row compute probability pij
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

def compute_prob_p(distance,sigmas): #Compute final matrix pi|j with optimal sigmas
    N = distance.shape[0]
    matrix = np.concatenate([[compute_proba_p_row(distance[i],sigmas[i],i) for i in range(N)]],axis = 0)
    return matrix

def negatif_squared_distance(X):  # Compute matrix of negative squared euclidean
    somme = np.sum(np.square(X), 1)
    distance_sqr = np.add(np.add(-2 * np.dot(X, X.T), somme).T, somme)
    return -distance_sqr

def compute_q(Y): # Compute low dimensional probabilities
    q = np.power(1. - negatif_squared_distance(Y), -1)
    np.fill_diagonal(q, 0.)  # qi,i
    return q / np.sum(q), q  # normalize in the entire matrix and return also the inverse distance

def p_conditional_to_joint(P):
    return (P + P.T) / (2. * P.shape[0]) # Sum of each row is not one anymore

def compute_joint_proba(data, target_perplexity):  # Compute joint probability from data and target perplexity
    N = data.shape[0]
    distances = np.square(euclidean_distances(data,data))
    sigmas = np.asarray([find_optimal_sigma(target_perplexity,distances[i],i) for i in range(N)])
    p_conditional = compute_prob_p(distances, sigmas)
    P = p_conditional_to_joint(p_conditional)
    return P,p_conditional

def tsne_grad(P, Q, Y, inv_distances): # Compute gradients
    pq_sub_exp = np.expand_dims(P - Q, 2)  # (pij −qij)
    y_sub = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)  # (yi −y)
    dist_exp = np.expand_dims(inv_distances, 2)  # (1+||yi−yj||2)**-1
    y_diffs_wt = y_sub * dist_exp
    # We sum over the j c.f formula gradient
    grad = 4. * (pq_sub_exp * y_diffs_wt).sum(1)  # (pij−qij)(yi−yj)(1+||yi−yj||2)−1
    return grad


def tsne_reduction(data, target_dim, target_perplexity, epochs = 200, lr=0.001, momentum=False, initialization = "random", mu = 0, std = 1): # Perform reduction
    N = data.shape[0]
    loss_history = []

    # Each initialization
    if initialization == "gaussian":
        Y = np.random.normal(mu, std, size=(N, target_dim))

    elif initialization == "pca":
        pca = PCA(n_components=target_dim)
        Y = pca.fit_transform(data)

    elif initialization == "laplace":
        Y = SpectralEmbedding(n_components=target_dim)
        Y = Y.fit_transform(data)

    else:
        Y = np.random.rand(N, target_dim)


    P,p_conditional = compute_joint_proba(data, target_perplexity) # P is the joint probas


    if momentum:
        Y_m2 = Y.copy()
        Y_m1 = Y.copy()


    for i in range(epochs):
        Q, inverse_distances = compute_q(Y)
        loss_history.append(loss_kl(P,Q))
        # print("Loss :", loss_kl(P, Q)) # If we want to see the current loss
        grads = tsne_grad(P, Q, Y, inverse_distances)

        Y = Y - lr * grads
        if momentum:
            Y += momentum * (Y_m1 - Y_m2)
            Y_m2 = Y_m1.copy()
            Y_m1 = Y.copy()

    return Y,loss_history,p_conditional

def loss_kl(matrix_p,matrix_q): # Compute kl loss given the two matrix probability
    loss = 0
    i = 0
    for p,q in zip(matrix_p,matrix_q): # kl on each row
        res = rel_entr(p,q)

        res[i] = 0 # Case where we have infinity because diag = 0 -> set to 0 to avoid infinity result
        i += 1
        loss += res
    return sum(loss)
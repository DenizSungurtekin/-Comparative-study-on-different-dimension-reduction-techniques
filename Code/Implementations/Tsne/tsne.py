import numpy as np

def negatif_squared_distance(X):
    # Compute matrix of negative squared euclidean
    # c.f https://stackoverflow.com/questions/37009647
    somme = np.sum(np.square(X), 1)
    distance_sqr = np.add(np.add(-2 * np.dot(X, X.T), somme).T, somme)
    return -distance_sqr

def compute_p_distribution(X, diag_zero=True):
    p_cond_ji = np.exp(X)
    if diag_zero:
        np.fill_diagonal(p_cond_ji, 0.) #pi|i = 0

    p_cond_ji = p_cond_ji + 1e-7  # To be able to compute entropie ( log_2(pi|j) if = 0 => problem )
    return p_cond_ji / p_cond_ji.sum(axis=1).reshape([-1, 1]) # Normalize each row


def compute_prob_matrix(distances, sigmas):
    return compute_p_distribution(distances / 2. * np.square(sigmas.reshape((-1, 1))))

def binary_search(compute_perplexity, target_perplexity, row,rows,max_iter=3000,tolerance=1e-5, upper=500.,lower=1e-15):
    for i in range(max_iter):
        sigma = (lower + upper) / 2.
        perplexity = compute_perplexity(sigma)
        if perplexity > target_perplexity:
            upper = sigma
        else:
            lower = sigma
        if np.abs(perplexity - target_perplexity) <= tolerance:
            break
        if i == max_iter - 1:
            rows.append(row) #To know for which row we do not have an ideal perplexity

    return sigma,rows

def calc_perplexity(prob_matrix):
    # Compute perplexity of each row
    entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1) #Carefull when proba = 0
    perplexity = 2 ** entropy
    return perplexity


def perplexity(distances, sigmas):
    #From distances and sigmas to perplexity vector (for each row)
    return calc_perplexity(compute_prob_matrix(distances, sigmas))


def find_optimal_sigmas(distances, target_perplexity):
    # We want the sigmas for each row giving the target perplexity
    sigmas = []
    rows = []
    for i in range(distances.shape[0]):
        compute_perplexity = lambda sigma: perplexity(distances[i:i+1, :], np.array(sigma))
        optimal_sigma,rows = binary_search(compute_perplexity, target_perplexity,i,rows)
        sigmas.append(optimal_sigma)

    print("Warning, an optimal standard deviation can't be found for rows: ",rows)
    return np.array(sigmas)

def p_conditional_to_joint(P):
    return (P + P.T) / (2. * P.shape[0])

def compute_joint_proba(data, target_perplexity): # Compute joint probability from data and target perplexity
    distances = negatif_squared_distance(data)
    sigmas = find_optimal_sigmas(distances, target_perplexity)
    p_conditional = compute_prob_matrix(distances, sigmas)
    P = p_conditional_to_joint(p_conditional)
    return P

def compute_q(Y):
    q = np.power(1. - negatif_squared_distance(Y), -1)
    np.fill_diagonal(q, 0.) # qi,i
    return q / np.sum(q), q # normalize in the entire matrix and return also the inverse distance

def tsne_grad(P, Q, Y, inv_distances):
    pq_sub_exp = np.expand_dims(P - Q, 2) #(pij −qij)
    y_sub = np.expand_dims(Y, 1) - np.expand_dims(Y, 0) # (yi −y)
    dist_exp = np.expand_dims(inv_distances, 2)  #(1+||yi−yj||2)**-1
    y_diffs_wt = y_sub * dist_exp
    # We sum over the j c.f formula gradient
    grad = 4. * (pq_sub_exp * y_diffs_wt).sum(1) #(pij−qij)(yi−yj)(1+||yi−yj||2)−1
    return grad


def tsne_reduction(data, target_dim, target_perplexity, iterations, lr = 0.01, momentum = False):

    # There is a lot of different initialisation method: The usual initialization routine for t-SNE is to start from a small random gaussian distribution,
    # and use “early exaggeration” for the first 100 or so iterations where the input probabilities are multiplied by 4

    Y = np.random.normal(0,1,size=(data.shape[0],target_dim)) * 4
    P = compute_joint_proba(data, target_perplexity)

    if momentum:
        Y_m2 = Y.copy()
        Y_m1 = Y.copy()

    for i in range(iterations):

        Q, inverser_distances = compute_q(Y)
        grads = tsne_grad(P,Q,Y,inverser_distances)

        Y = Y - lr*grads
        if momentum:
            Y += momentum * (Y_m1 - Y_m2)
            Y_m2 = Y_m1.copy()
            Y_m1 = Y.copy()

    return Y

N = 40
M = 20
data = np.random.rand(N,M)
print("Orginal data: ",data)
print("")
target_perplexity = 5
Y = tsne_reduction(data,2,5,100)
print("")
print("Embeed data: ",Y)
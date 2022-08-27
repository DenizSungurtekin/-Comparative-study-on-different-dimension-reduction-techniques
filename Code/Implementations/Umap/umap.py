from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import scipy.optimize as op
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding

def computeProbMatrix(sigmas,dist,rhos): #Compute pi|j, sigmas 1d vector containing sgd for each point
    N = len(dist)
    prob_matrix = np.asarray([[computeProba(dist[i][j],sigmas[i],rhos[i]) if i!=j else 1 for j in range(N)] for i in range(N)])
    return prob_matrix

def computeProba(dist,sigma,rho):
    return np.exp(-((dist-rho)/sigma))

def p_conditional_to_joint(P):
    return (P + P.T) - np.multiply(P,P.T)

def compute_k(P_row): #return k for a row
    return 2**P_row.sum()

def computeProbEmbedMatrix(Y,a,b):
    prob_matrix = np.power(1 + a * np.square(euclidean_distances(Y, Y))**(2*b),-1)
    return prob_matrix


def computeGrad(P, Y,a,b):
    y_diff = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
    inv_dist = np.power(1 + a * np.square(euclidean_distances(Y, Y))**b, -1)
    Q = np.dot(1 - P, np.power(0.001 + np.square(euclidean_distances(Y, Y)), -1))
    np.fill_diagonal(Q, 0)
    Q = Q / np.sum(Q, axis = 1, keepdims = True)
    fact=np.expand_dims(a*P*(1e-8 + np.square(euclidean_distances(Y, Y)))**(b-1) - Q, 2)
    return 2 * b * np.sum(fact * y_diff * np.expand_dims(inv_dist, 2), axis = 1),Q


def compute_p_row(dist,sigma,rho,normalization):
    if normalization:
        probas = np.asarray([np.exp(-((dist[i]-rho)/sigma)) for i in range(len(dist))])
        somme = probas.sum()
        probas = probas/somme
        return probas
    else:
        return np.asarray([np.exp(-((dist[i]-rho)/sigma)) for i in range(len(dist))])

def find_optimal_sigma(target_k,rho,dist_row,normalization,max_iterations = 30,tolerance = 1e-5,upper = 500.,low = 0.): #Binary search to find sigma with target k (nb of nearest neighbor)

    for i in range(max_iterations): # if k big then sigma larger
        current_sigma = (upper+low)/2
        p_row = compute_p_row(dist_row,current_sigma,rho,normalization)
        p_row = p_conditional_to_joint(p_row)
        current_k = compute_k(p_row)
        if np.abs(current_k-target_k) < tolerance:
            break
        else:
            if current_k<target_k:
                low = current_sigma
            else:
                upper = current_sigma

    return current_sigma

def find_best_a_b(min_dist): # On veut trouver a et b utilise dans le calcul de q. Idée est qu on a (1+a( dis(y)**2b)**-1 ~= 1 si y < min dist ou e**-(y) - min_dist si y > min dist => on calcule cela pour obtenir les valeurs sur y pour trouver a et b avec moindre carrée
    x = np.linspace(0,10,500) # Upper bound choisis selon distance maximum entre les y (pas plus grd que 10 normalement)
    output = []
    for el in x:
        if el<=min_dist:
            output.append(1)
        else:
            output.append(np.exp(-el)-min_dist)

    f = lambda x,a,b: 1/(1+a*np.power(x,2*b))
    parameter,_ = op.curve_fit(f,x,output)

    # m = lambda x:1/(1+parameter[0]*np.power(x,2*parameter[1])) to plot the curve
    # l = [m(el) for el in x]
    # plt.plot(x,l)
    # plt.show()

    return parameter[0],parameter[1] #a,b


def umap_reduction(data,min_dist,target_dim,target_k,normalization = False,epochs = 200,lr = 0.02,initialization = "random", mu = 0, std = 1,SGD = False):

    loss_history = []
    a,b = find_best_a_b(min_dist)
    N = data.shape[0]
    dist = np.square(euclidean_distances(data, data))  # Matrice de distance au carré avec norme 2 -> euclidian
    rhos = [sorted(dist[i])[1] for i in range(N)]  # Distance avec le voisin le plus proche de chaque point p_i
    sigmas = np.asarray([find_optimal_sigma(target_k,rhos[i],dist[i],normalization) for i in range(len(dist))])
    P_cond = computeProbMatrix(sigmas,dist,rhos)
    P = p_conditional_to_joint(P_cond)
    # P = P / np.sum(P, axis = 1, keepdims = True) # If we want to normalize but not done in umap

    if initialization == "gaussian":
        Y = np.random.normal(mu, std, size=(data.shape[0], target_dim))

    elif initialization == "pca":
        pca = PCA(n_components=target_dim)
        Y = pca.fit_transform(data)

    elif initialization == "laplace":
        Y = SpectralEmbedding(n_components=target_dim)
        Y = Y.fit_transform(data)

    else:
        Y = np.random.rand(N, target_dim)

    if SGD:
        for i in range(epochs):
            random_index = np.random.randint(0,len(data))
            grad,Q = computeGrad(P,Y,a,b)
            Y[random_index] = Y[random_index] - lr*grad[random_index]
            loss_history.append(cross_entropy(P,Y,a,b))
    else:
        for i in range(epochs): #Not SGD, standard gradient descend
            grad,Q = computeGrad(P,Y,a,b)
            Y = Y - lr*grad
            loss_history.append(cross_entropy(P,Y,a,b))
    return Y,loss_history,P_cond


def cross_entropy(P,Y,a,b):
    Q = np.power(1 + a * np.square(euclidean_distances(Y, Y))**b, -1)
    cross = - P * np.log(Q + 0.01) - (1 - P) * np.log(1 - Q + 0.01)
    return np.sum(cross) / (P.shape[0]*P.shape[1])
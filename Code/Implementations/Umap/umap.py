from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

N = 100
M = 20

a = 1
b = 1 # UMAP finds a and b from non-linear least-square fitting to the piecewise function with the min_dist hyperparameter:

min_dist = 0.25 # Hyperparametre influant sur a et b mais ici initialise à 1
embed_dim = 2 #hyperparam
#n_epoch -> hyperparam
#n -> hyperparam , the number of neighbors to consider when approximating the local metric; utilise dans spectral embedding

X_train = np.random.rand(N,M)
dist = np.square(euclidean_distances(X_train, X_train)) #Matrice de distance au carré avec norme 2 -> euclidian
rhos = [sorted(dist[i])[1] for i in range(N)] # Distance avec le voisin le plus proche de chaque point p_i

#Calcul de l'entropie comme la somme des probas de chaque cellule de la matrice de distance pour choisir k = 2**Entropy pour definri sigmnas TODO
sigmas = np.random.rand(N)

## NORMALEMENT Y INITIALISE AVEC SPECTRAL EMBEDING A VOIR PLUS EN DETAIL
Y = np.random.rand(N,embed_dim)

def computeProbMatrix(sigmas,dist,rhos): #Compute pi|j, sigmas 1d vector containing sgd for each point
    N = len(dist)
    prob_matrix = np.asarray([[computeProba(dist[i][j],sigmas[i],rhos[i]) if i!=j else 1 for j in range(N)] for i in range(N)])
    return prob_matrix

def computeProba(dist,sigma,rho):
    return np.exp(-((dist-rho)/sigma)) #Symetrique dans notre cas pas besoin de faire pij = pi|j + pj|i - pi|j*pj|i ATTENTION FORMULE A MODULE


def computeProbEmbedMatrix(Y):
    prob_matrix = np.power(1 + a * np.square(euclidean_distances(Y, Y))**b, -1)
    # euclidean_distances(Y,Y) ex: ligne 1 contient distance de y_1 aux autre y -> dist[1][2] -> distance y1-y2
    return prob_matrix

# Optimization

# def cross_entropy(ProbMatrix,EmbedProbMatrix): # +0.01 pour eviter 0 -> On s'interesse plus au gradient pour mettre a jour Y
#     return - ProbMatrix * np.log(EmbedProbMatrix + 0.01) - (1 - ProbMatrix) * np.log(1 - EmbedProbMatrix + 0.01)

def computeGrad(ProbaMatrix,Y): #Derivative of cross entropy

    y_differences = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
    embedProbMatrix = computeProbEmbedMatrix(Y)
    Q = np.dot(1 - ProbaMatrix, np.power(0.001 + np.square(euclidean_distances(Y, Y)), -1)) # Q contient
    np.fill_diagonal(Q, 0)
    Q = Q / np.sum(Q, axis = 1, keepdims = True) # Normalization pas dans u_map mais peut donner meilleur resultat
    fact = np.expand_dims(a*ProbaMatrix*(1e-8 + np.square(euclidean_distances(Y, Y)))**(b-1) - Q, 2)
    return 2 * b * np.sum(fact * y_differences * np.expand_dims(embedProbMatrix, 2), axis = 1)


def main(X_train,min_dist,embed_dim,a=1,b=1,sigmas=sigmas,epochs = 100,learning_rate = 0.01):
    dist = np.square(euclidean_distances(X_train, X_train))  # Matrice de distance au carré avec norme 2 -> euclidian
    rhos = [sorted(dist[i])[1] for i in range(N)]  # Distance avec le voisin le plus proche de chaque point p_i
    proba_matrix = computeProbMatrix(sigmas,dist,rhos)
    y = np.random.rand(N,embed_dim)
    for i in range(epochs):
        y = y - computeGrad(proba_matrix,y)
    return y


print(main(X_train,min_dist,embed_dim))
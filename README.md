# Cachier des charges: 
  Implémentation tsne: 
  
  - 23.02.2022: Calcul de la matrice contenant les distributions normalisées p_j|i pour un dataset donné avec les sigmas spécifiés.  
  - 25.02.2022: 1: Calcul de la matrice contenant les distributions normalisées q_j|i pour un manifold.    
                2: Calcul du gradient avec loss KL et mis à jour du manifold.  
                3: Première implémentation "fonctionnelle" simple de t-sne avec une méthode d'initialisation du manifold et des variances aléatoires.  
  - 19.03.2022: 1: Transition Pytorch à numpy.  
                2: Calcul des sigmas optimaux pour une perplexité donnée.
                3: Initalisations des Y avec sampling d'une petite distribution gaussienne.  
                4: Ajout du momentum dans le gradient descent. (Optionnel)
                5: Ajout de la conversion de la matrice des probabilités conditionnel à jointe.  
                
  - 20.04.2022: Implémentation des différents modes d'initialisation de l'embedding: "Random-Gaussian-Laplace/Spectral Embedding-Pca"
  - 21.04.2022: Implémentation de la génération de dataset swissroll et l'automatisation des plots selons l'epochs, learning_rate, ecart type du bruit ajouté comme 4ème dimension au dataset, perplexité, threshold permettant de définir la proba minimal pour considérer un point comme étant un voisin (Estimation de la densité), + plot diagram de shepard + plot loss + sauvegarde des plots.
  - 23.04.2022: Ajout des 256 images pour run de 4D à 3D + Ajout des 256 images pour run de 4D à 2D. Avec comme parametres: epochs = 1000, lr = 100, perplexité = [5,15,25,50], stds = [1,5,10,20], threshold = 0.05.
  
   
               
    
  
    
  
  Implémentation Umap:
  
  - 03.03.2022:  1: Calcul de la matrice contenant les distributions p_j|i pour un dataset donné avec les sigmas spécifiés.   
               2: Calcul de la matrice contenant les distributions q_j|i   
               3: Calcul du gradient avec loss cross-entropy et mis à jour du manifold.   
               4: Première implémentation "fonctionnelle" simple de umap avec une méthode d'initialisation du manifold aléatoire et des variances aléatoires. Comme pour tsne la                      manière dont on initialise le manifold et les sigmas utilisés sont les points "sensibles" nécessitant plus de reflexion.  
  
  - 24.03.2022:  1: Calcule des sigmas optimaux selon le k target fixé par l'utilisateur  
                 2: Calcule des parametres a et b selon la distance minimal fixé par l'utilisateur. (Obtenu par moindre carré)  
                 3: Ne manque plus que d'essayer différente méthode d'initialisation des y.  
  - 20.04.2022: Implémentation des différents modes d'initialisation de l'embedding: "Random-Gaussian-Laplace/Spectral Embedding-Pca"
  - 21.04.2022: Implémentation de la génération de dataset swissroll et l'automatisation des plots selons l'epochs, learning_rate, ecart type du bruit ajouté comme 4ème dimension au dataset, k, threshold permettant de définir la proba minimal pour considérer un point comme étant un voisin (Estimation de la densité), + plot diagram de shepard + plot loss + sauvegarde des plots.
  - 27.04.2022: Ajout des 256 images pour run de 4D à 3D + Ajout des 256 images pour run de 4D à 2D. Avec comme parametres: epochs = 1000, lrs = [0.02,0.02,0.02,0.002], k_values = [2,15,25,50], stds = [1,5,10,20], threshold = 0.1, min_dist = 0.1.
               
    
                      

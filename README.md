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
                6: Ne manque plus que d'essayer différente méthode d'initialisation des y.  
    
  
    
  
  Implémentation Umap:
  
  - 03.03.2022:  1: Calcul de la matrice contenant les distributions p_j|i pour un dataset donné avec les sigmas spécifiés.   
               2: Calcul de la matrice contenant les distributions q_j|i   
               3: Calcul du gradient avec loss cross-entropy et mis à jour du manifold.   
               4: Première implémentation "fonctionnelle" simple de umap avec une méthode d'initialisation du manifold aléatoire et des variances aléatoires. Comme pour tsne la                      manière dont on initialise le manifold et les sigmas utilisés sont les points "sensibles" nécessitant plus de reflexion.  
  
  - 24.03.2022:  1: Calcule des sigmas optimaux selon le k target fixé par l'utilisateur  
                 2: Calcule des parametres a et b selon la distance minimal fixé par l'utilisateur. (Obtenu par moindre carré)  
                 3: Ne manque plus que d'essayer différente méthode d'initialisation des y.  
               
    
                      

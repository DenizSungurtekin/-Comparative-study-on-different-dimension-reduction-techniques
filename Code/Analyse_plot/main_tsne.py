from Code.Implementations import tsne
from Code.Analyse_plot import func_tools
from Code.DataHandler import dataset_generations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# Generate Data
size = 200
data,color_swiss = dataset_generations.make_swiss_roll(size)

# Save and plot Original data
func_tools.scatter3d(data, color=color_swiss,plot=False,title="Original Data")
func_tools.save_plot_tsne("Original")
plt.show()



# Function used to do the EXPERIMENT 1 under t-sne. The plots are showed and saved if desired
def all_plots(data,target_dim,lr,epochs,save = True,show = True): # If saves we save plots in plot/tsne, if show we shows the plots one by one
    modes = ["random","gaussian","pca","laplace"]
    colors = ["blue","red","orange","black"]
    stds = [1,5,10,20]
    perplexities = [5,15,25,50]
    threshold = 0.05 # Threshold which define the minimum probability to be considered as a neighbor
    size = (data.shape[0],1)

    if target_dim == 2 or target_dim == 3:
        for mode,color in zip(modes,colors):
            for std in stds:
                for perplexity in perplexities:
                    noise = np.random.normal(0, std, size=size)
                    data2 = np.append(data, noise, axis=1)  # add noise to swiss roll

                    distances = np.square(euclidean_distances(data2,data2))
                    Y, loss, conditional_probas = tsne.tsne_reduction(data2, target_dim, perplexity, epochs=epochs, initialization=mode, lr=lr)
                    string = mode+"std"+str(std)+"perplex"+str(perplexity)

                    #Plot example of one row distribution
                    func_tools.draw_dist_neighboorhood(conditional_probas,0)

                    # Plot loss
                    plt.plot(loss)
                    plt.xlabel("Iterations")
                    plt.ylabel("Loss")
                    plt.title("Loss with mode = " + str(mode)+ " std = "+str(std)+" Perplexity = "+str(perplexity))
                    save_name_loss = string+"loss"
                    if save:
                        func_tools.save_plot_tsne(save_name_loss)
                    if show:
                        plt.show()

                    # Plot Embedding
                    plt.clf()
                    save_name_embeed = string+"tsne"
                    if target_dim == 3:
                        func_tools.scatter3d(Y, color=color_swiss,plot=False,title="Embedding with "+string)
                    elif target_dim == 2:
                        func_tools.scatter2d(Y, color=color_swiss, plot=False, title="Embedding with " + string)

                    if save:
                        func_tools.save_plot_tsne(save_name_embeed)
                    if show:
                        plt.show()


                    #Plot Shepard diagram
                    plt.clf()
                    save_name_diagram = string+"diagram"
                    distances_embeed = np.square(euclidean_distances(Y, Y))
                    func_tools.shepard_diag(distances, distances_embeed,plot = False)
                    if save:
                        func_tools.save_plot_tsne(save_name_diagram)
                    if show:
                        plt.show()
                    plt.clf()


                    #Plot density of conditional matrix (Number of neigboor for each sample)
                    save_name_density= string+"density"
                    func_tools.plot_conditional_density(conditional_probas, threshold,plot=False)
                    if save:
                        func_tools.save_plot_tsne(save_name_density)
                    if show:
                        plt.show()
                    plt.clf()
    else:
        print("Error target dimension need to be 2D or 3D...")



# Call plots functions with desired hyper parameters
#-------4D To 3d Reduction---------

epochs = 1000
lr = 100
target_dim = 3


## Different call of the plot function: CHOOSE ONE

all_plots(data,target_dim,lr,epochs) #To show and save
# all_plots(data,target_dim,lr,epochs,show = False) # Only save
# all_plots(data,target_dim,lr,epochs,save = False) # Only show


#-------4D To 2d reduction---------
epochs = 1000
lr = 100
target_dim = 2

all_plots(data,target_dim,lr,epochs) #To show and save
# all_plots(data,target_dim,lr,epochs,show=False) # Only save
# all_plots(data,target_dim,lr,epochs,save = False) # Only show

# ---------------------------

## --- Stability analyse EXPERIMENT 2 ----

def checkList(l1,l2): # Check if element of l1 are not in L2
    for el in l1:
        if el in l2:
            return True
    return False

indexs= False # Used to have the same random indexes with diferent dataset and methods
def compute_stability_matrix(s1,s2,data_original,data,indexs): # Compute configuration score defined in experiment 2
    # s1,s2 the number of pivot points

    N,_ = data.shape
    i = 0
    message = True

    if not indexs:
        indexs1 = [np.random.randint(0,N) for i in range(s1)]
        indexs2 = [np.random.randint(0,N) for i in range(s2)]

        while checkList(indexs1,indexs2): # Be sure that indexes are different
            if i > 1000 and message == True:
                print("Warning maybe in an infinite loop")
                message = False

            indexs1 = [np.random.randint(0, N) for i in range(s1)]
            indexs2 = [np.random.randint(0, N) for i in range(s2)]
            i += 1
    else:
        indexs1 = indexs[0]
        indexs2 = indexs[1]

    data1_original = data_original[indexs1]
    data2_original = data_original[indexs2]
    dist_original = np.square(euclidean_distances(data1_original, data2_original))

    meanRow_original = np.asarray([np.mean(dist_original[i]) for i in range(s1)])

    data1 = data[indexs1]
    data2 = data[indexs2]

    dist = np.square(euclidean_distances(data1, data2))
    meanRow = np.asarray([np.mean(dist[i]) for i in range(s1)])

    stability_score = np.mean(np.abs(meanRow_original - meanRow))
    return stability_score,[indexs1,indexs2]

def map(vector): # Mapping to [0.1]

    res = []
    min = np.min(vector)
    max = np.max(vector)

    for el in vector:
        res.append(1 - (el-min)/(max-min))

    return np.asarray(res)

# Implementation of Experiment 2
def checkStability_tsne(name="Default",data=data,s1=10,s2=100,target_dim=2,perplexities=[2,15,35,50],epochs=500,modes=["random","gaussian","laplace","pca"],lr=100):
    global indexs
    scores = []
    infos = [] # Contain parameters value
    for mode in modes:
        for perplexity in perplexities:

            Y,_,_ = tsne.tsne_reduction(data, target_dim, perplexity, epochs=epochs, initialization=mode, lr=lr)
            score,indexs = compute_stability_matrix(s1,s2,data,Y,indexs)
            info = "p= ",perplexity, "mode = ",mode
            scores.append(score)
            infos.append(info)


    scores = map(scores) # Si score -> 1 alors bonne stabilité car petite mean distance sinon 0
    return np.asarray(list(zip(scores,infos)))
from Code.Implementations.Umap import umap
from Code.Analyse_plot import func_tools
from Code.Implementations.DataHandler import generations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# Generate Data, plot and save original data
size = 200
data,color_swiss = generations.make_swiss_roll(size)

#Save and plot
# func_tools.scatter3d(data, color=color_swiss,plot=False,title="Original Data")
# # func_tools.save_plot_umap("Original")
# plt.show()

def all_plots(data,target_dim,epochs,save = True,show = True): # If saves we save plots in plot/tsne, if show we shows the plots one by one
    modes = ["random","gaussian","laplace","pca"]
    lrs = [0.02,0.02,0.02,0.002] # Less for pca because it s directly well reduced so with lr too big we increase the loss
    colors = ["blue","red","orange","black"]
    stds = [1,5,10,20]
    k_values = [2,15,25,50] #n_neighbors Default = 15 -> 2 to a quarter of the data here size / 4 = 200/4 = 50
    min_dist = 0.1 # default value from official 0.1
    threshold = 0.1 # Threshold which define the minimum probability to be considered as a neighbor
    size = (data.shape[0],1)

    if target_dim == 2 or target_dim == 3:
        for mode,color,lr in zip(modes,colors,lrs):
            for std in stds:
                for k in k_values:

                    noise = np.random.normal(0, std, size=size)
                    data2 = np.append(data, noise, axis=1)  # add noise to swiss roll

                    distances = np.square(euclidean_distances(data2,data2))
                    Y, loss, conditional_probas = umap.umap_reduction(data2,min_dist, target_dim, k, epochs=epochs, initialization=mode, lr=lr,SGD=False)
                    string = mode+"std"+str(std)+"K"+str(k)

                    # #Plot example of row distribution
                    # for i in range(10):
                    #     func_tools.draw_dist_neighboorhood(conditional_probas,i)

                    # Plot loss
                    plt.plot(loss)
                    plt.xlabel("Iterations")
                    plt.ylabel("Loss")
                    plt.title("Loss with mode = " + str(mode)+ " std = "+str(std)+" K = "+str(k))
                    save_name_loss = string+"loss"
                    if save:
                        func_tools.save_plot_umap(save_name_loss)
                    if show:
                        plt.show()

                    # Plot Embedding
                    plt.clf()
                    save_name_embeed = string+"umap"
                    if target_dim == 3:
                        func_tools.scatter3d(Y, color=color_swiss,plot=False,title="Embedding with "+string)
                    elif target_dim == 2:
                        func_tools.scatter2d(Y, color=color_swiss, plot=False, title="Embedding with " + string)

                    if save:
                        func_tools.save_plot_umap(save_name_embeed)
                    if show:
                        plt.show()


                    #Plot Shepard diagram
                    plt.clf()
                    save_name_diagram = string+"diagram"
                    distances_embeed = np.square(euclidean_distances(Y, Y))
                    func_tools.shepard_diag(distances, distances_embeed,plot = False)
                    if save:
                        func_tools.save_plot_umap(save_name_diagram)
                    if show:
                        plt.show()
                    plt.clf()


                    #Plot density of conditional matrix (Number of neigboor for each sample)
                    save_name_density= string+"density"
                    func_tools.plot_conditional_density(conditional_probas, threshold,plot=False,umap = True)
                    if save:
                        func_tools.save_plot_umap(save_name_density)
                    if show:
                        plt.show()
                    plt.clf()
    else:
        print("Error target dimension need to be 2D or 3D...")




# Main: Run different configs and plot results

#-------4D To 3d plot---------
# TSNE: some parameters to fix before plots
# epochs = 1000
# target_dim = 3
#
#
# # Different call of the plot function
# # all_plots(data,target_dim,epochs) #To show and save
# all_plots(data,target_dim,epochs,show = False) # Only save
# # all_plots(data,target_dim,epochs,save = False) # Only show


#-------4D To 2d---------
## TSNE: some parameters to fix before plots
# epochs = 1000
# target_dim = 2


# Different call of the plot function
# all_plots(data,target_dim,epochs) #To show and save
# all_plots(data,target_dim,epochs,show=False) # Only save
# all_plots(data,target_dim,epochs,save = False) # Only show

# ---------------------------

## --- Stability analyse ----

def checkList(l1,l2): # Check if element of l1 are not in L2
    for el in l1:
        if el in l2:
            return True
    return False

## Stability matrix dist and mean by row
indexs= False
def compute_stability_matrix(s1,s2,data_original,data,indexs):
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

    # print(i)
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

def map(vector):

    res = []
    min = np.min(vector)
    max = np.max(vector)

    for el in vector:
        res.append(1 - (el-min)/(max-min))

    return np.asarray(res)

# Check general stability of different initialization method/Noise intensity/No noise/k_values
def checkStability(name,data=data,s1=5,s2=200,target_dim=2,ks=[2,15],epochs=500,modes=["random","gaussian","laplace","pca"],lr=0.02,stds=[1,5]):
    global indexs
    scores = []
    size = (data.shape[0], 1)
    min_dist = 0.1

    for mode in modes:
        for k in ks:
            for std in stds:

                noise = np.random.normal(0, std, size=size)
                data2 = np.append(data, noise, axis=1)  # add noise t

                Y,_,_ = umap.umap_reduction(data2,min_dist, target_dim, k, epochs=epochs, initialization=mode, lr=lr,SGD=False)
                score,indexs = compute_stability_matrix(s1,s2,data,Y,indexs)
                scores.append(score)


    # # Save data
    # name1 = savename + "DistMatrix.txt"
    # name2 = savename + "MeanRow.txt"
    #
    # name3 = savename + "DistMatrix.npy"
    # name4 = savename + "MeanRow.npy"
    #
    # np.savetxt(name1, tot_dist, fmt="%s")
    # np.savetxt(name2, meanRow, fmt="%s")
    #
    # with open(name3,"wb") as f:
    #     np.save(f,tot_dist)
    #
    # with open(name4,"wb") as f:
    #     np.save(f,meanRow)

    return map(scores) # Si score -> 1 alors bonne stabilité car petite mean distance sinon 0

# # Run test first over all modes then mode by mode to see the stability
# print(checkStability("generalTest"))
# print(checkStability("random",modes=["random"]))
# checkStability("gaussian",modes=["gaussian"])
# checkStability("laplace",modes=["laplace"])
# checkStability("pca",modes=["pca"])
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


# Main


#-------4D To 3d---------
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
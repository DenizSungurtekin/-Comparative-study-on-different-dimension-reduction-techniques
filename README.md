## Comparative study on UMAP and t-SNE

![unige_csd](https://user-images.githubusercontent.com/43375040/190927447-6ee858d9-0233-4241-8ba3-a1e851ceecf4.png) 

Here you can find my **computer science master project**, a comparative study on **UMAP** and **t-SNE** accompanied by a personal and simplified implementation of these techniques.

You can find the following elements:
  - **Code**: Contain the entire code behind the project. The architecture is described just below.
  - **Oral Presentation.pdf/ppt**: The slides used during the oral examination.
  - **paper.pdf**: My paper, the comparative study on UMAP and t-SNE.
  
Here is a detailed description of the implementation:

Under the Code folder, you will find three folders:
  - **Analyse_plot**: This contains the plots and the code used for my experiments (c.f paper).
  - **DataHandler**: Folder containing dataset_generations.py, a simple code used to generate the datasets used in this work.
  - **Implementations**: Folder containing tsne.py and umap.py, my implementations of the corresponding reduction technique.
  
Under the Analyse_plot folder, you will find the following folders/files:
  - **Plots**: Contain the plots generated by all the experiments.
  - **evaluation.py**: Implementation of experiments 0 and 3.
  - **evaluation_measure.py**: Implementation of the metrics used to evaluate the stability of the methods.
  - **func_tools.py**: File that contains useful functions to make the plots.
  - **main_tsne.py**: Implementation of experiments 1 and 2 for t-SNE.
  - **main_umap.py**: Implementation of experiments 1 and 2 for UMAP.
  
Under the Plots folder, you will find the following folders:
  - **global**: Folder that contains plots of spearman's rho and stress values.
  - **tsne**: Folder that contains plots of t-SNE (Experiments 1,2, and 3).
  - **umap**: Folder that contains plots of t-SNE (Experiments 1,2, and 3).

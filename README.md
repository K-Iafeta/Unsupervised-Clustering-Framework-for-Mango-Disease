# Unsupervised-Clustering-Framework-for-Mango-Disease
Project script initially utilized in high-performance computing linux environment for extracting feature embeddings and run multiple clustering analyses on a large ~25,000 image dataset.

**Note**: All scripts were run in a High-Performance Computer (HPC) linux environment made available from QUT. The memory used for the analysis was **"ncpus = 8, mem=32GB"**

The first script that needs to be run is the "SamplesToCSV.py" which generates the .csv file necessary as the input for the feature extractors. The feature extractors assume that there is a folder with the image dataset present in the same workspace as these scripts. 

The Repository currently contains a folder of the feature extraction code used to extract the feature embeddings of a ~25,000 image dataset. The first script that needs to be run is the Frozen ViT-B/16 feature extraction code because the dataset split is only present in that script and the generated .csv files of the train/val/test split is then used directly for the 3 other feature extractors. After the Frozen ViT script is run, then the other 3 can be run to extract the feature embeddings

After the feature embeddings are extracted, the "mango_embedding_cluster_evaluation_new.py" script is run to get the analysis results for the four different clustering analyses.

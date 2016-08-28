## K-Means classifier


### Structure
* /data: contains all the data files (orginal segmentation file modified to remove junk in first few lines, switch label position to last column)
* /plots: plots for the mentioned data-sets (with original label + K-mean output for that value of 'k')
* main.py : main file to be run. Run `python main.py <path_to_file> <value_of_k>`. For example, `python data/segmentation.txt 3`.
* my_kmeans.py : contains implementation of k-means (and GMM)
* my_kmeans.py : contains implementation of RI (code borrowed from sklearn)


### Additional Info
* Make sure that the system on which this code is run has 'matplotlib' installed on it (in addition to sklearn and numpy).
* Uncomment the last part in my_kmeans.py to generate error/iteration plots for every run.
* Based on observation, a single run of k-means has only been run for 10 updates (as running it for anything more that that did not lead to any decrease in error. Usually, the error stops dropping after 4-5 runs).
* Implementation of RI slighly borrowed from https://github.com/janelia-flyem/gala/blob/a55ceaa5c757366a26946b2c6ad0dd547966876b/gala/evaluate.py (in addition to base code from sklearn)
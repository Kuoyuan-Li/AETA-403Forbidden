import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from config import *


def concat_eq_data ():
    fileName_list = os.listdir(Data_Folder_Path)  # basic path
    re_eq = re.compile(r'EC_(\d){8}-(\d){8}.csv')  # re to find all earthquake data files
    eq_fileName_list = [f for f in fileName_list if re.search(re_eq, f)]  # find eq data files
    eq_fileName_list = [Data_Folder_Path + eq_file for eq_file in eq_fileName_list]  # add path
    eqlst = open(Eq_list_path,"w")  # merge all eq data to eqlst.csv
    if (len(eq_fileName_list))== 0: # no eq data file being found
        eqlst.write(eq_header)
        eqlst.close()
    elif (len(eq_fileName_list))== 1:  # only one eq data file being found
        for line in open(eq_fileName_list[0]):
            eqlst.write(line)
        eqlst.close()
    else:  # multiple eq data files being found
        for line in open(eq_fileName_list[0]):
            eqlst.write(line)
        # now the rest eq files:
        for i in range(1, len(eq_fileName_list)):
            eq_data = open(eq_fileName_list[i])
            eq_data.__next__() # skip the header
            for line in eq_data:
                eqlst.write(line)
            eq_data.close()
        eqlst.close()

def eq_cluster_Kmeans():
    eq_data = pd.read_csv(Eq_list_path)
    eq_data = eq_data[['Longitude','Latitude','Magnitude']]
    eq_loc = eq_data[['Longitude','Latitude']].to_numpy()  # use location to cluster earthquakes
    clusters = KMeans(n_clusters=8).fit_predict(eq_loc)  # no weight cluster
    # visualizing the results
    u_labels = np.unique(clusters)
    for i in u_labels:
        plt.scatter(eq_loc[clusters == i, 0], eq_loc[clusters == i, 1], label=i)
    plt.legend()
    plt.show()



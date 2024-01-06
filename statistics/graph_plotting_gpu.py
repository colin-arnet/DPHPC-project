#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing the required module
import glob
import itertools
import math
import os
import re
import shutil
import sys
import zipfile
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from pandas.core.frame import DataFrame
from scipy.stats import t

#################################################################################################################
# Seven summary graphs, either comparing the performance of MPI_Kokkos vs. MPI_OMP, or
# Kokkos(naive) vs. KokkosDAG vs. OMP.
# Summary graphs are initialized in the function 'plot_by_size' (after having created all individual graphs)
# and are plotted by the function 'plot_summary_graphs', which is called at the end of generate_statistics_by_type
##################################################################################################################

PLOTS_DIR = 'plots'
EXEC_TIMES_DIR = 'execution_times'
SPEEDUP_DIR = 'speedups'
CI_DIR = 'confidence_intervals'

graphs = {}


def generate_for_zip(zippath: str, outputzip: str):
    """ convenience-wrapper around generate_statistics_by_type to deal with zipped results """
    with zipfile.ZipFile(zippath, 'r') as zip_ref:
        dir_name = zippath.split('/')[-1].split('.')[0]
        zip_ref.extractall(dir_name)
        print("{} contains {} files".format(
            dir_name, len(os.listdir(dir_name))))
        generate_statistics_by_type(dir_name, outputzip)
        shutil.rmtree(dir_name)


def generate_statistics_by_type(path: str, outputzip: str):
    """ Groups measurements by matrix-size and implementation-type, and plots the
        mean execution times
    """
    # setup temporary plots directory
    if os.path.exists(PLOTS_DIR) and os.path.isdir(PLOTS_DIR):
        print("Found old temporary plots directory, deleting...")
        shutil.rmtree(PLOTS_DIR)
    os.mkdir(PLOTS_DIR)
    # setup ConfidenceIntervals directory
    os.mkdir(os.path.join(PLOTS_DIR, CI_DIR))

    df_list = []
    

    for result_file in os.scandir(path):
        if result_file.is_file():
            print("Reading file", result_file.name)
            df_list.append(pd.read_csv(result_file, skiprows=4,
                           sep='\s+', engine='python'))
  
    for df in df_list:
        # split into problem size
        if 'matrix_size' in df.columns:
            size_col = 'matrix_size'
        else:
            size_col = 'w'
        plot_by_size(df, size_col)

    # plot the summary graphs, result is saved in /dphpc_project/Plots/SummaryPlots
    plot_execution_time_summary_graphs(size_col)

    # Zip plots dir (not overwriting any existing)
    no_of_files = len(list(filter(lambda fn: outputzip in fn,
                      os.listdir('.'))))  # add unique id at end of filename
    shutil.make_archive("{}_{}".format(
        outputzip, no_of_files), 'zip', PLOTS_DIR)

    # Delete temporary plots dir
    shutil.rmtree(PLOTS_DIR)


#####################################
# ******** Plot functions **********
#####################################

def plot_by_size(df: DataFrame, size_col: str):
    """ This function plots process-mean execution time graphs for each implementation
        Datapoints are stored into the summary graphs, which are needed later
    """

    # type of current df
    t = df['type'].iloc[0]

    print("\nGenerating plot for {}...".format(t))

    # list of coordinate points (x,y) where x = size and y = mean execution time
    points = []
    # split into size
    grouped_df = df.groupby(size_col)
    for size in df[size_col].unique():
        df2 = grouped_df.get_group(size)
        print("----- Processing for size {} -----".format(size))
        m, df2, ci, diff = preprocess(df2)
        print("Mean execution time: {}".format(m))
        print("confidence interval: {}".format(ci))
        points.append(((size, m), diff))

    # split points into x and y coordinates
    points.sort(key=lambda p: p[0])
    pts, diffs = zip(*points)
    x, y = zip(*pts)
    # plotting the points
    plt.plot(x, y, color='green', linestyle='dashed', linewidth=3,
             marker='o', markerfacecolor='blue', markersize=10)

    # create an index for each tick position
    x_ticks = x
    x_labels = x
    #add x-axis values to plot
    plt.xticks(ticks=x_ticks, labels=x_labels)
    # styling
    plt.grid(True)
    plt.tick_params(grid_alpha=0.5)
    _, top_y = plt.ylim()
    margin = 0.1
    plt.ylim(bottom=0, top=(1 + margin)*top_y)

    # naming the x axis
    plt.xlabel(size_col)

    # naming the y axis
    plt.ylabel('arithmetic mean execution time (ms)')

    # giving a title to my graph
    title = t + ' execution time'
    plt.title(title)

    filename = '{}/{}.png'.format(PLOTS_DIR, t)
    print("--> Storing results to {}".format(filename))

    # function to save and show the plot
    plt.savefig("./{}".format(filename))
    plt.clf()
    
    # generate CI plots
    print("\nGenerating confidence interval plots...")
    
    diffs = np.array(diffs)
    
    plt.errorbar(x, y, yerr=diffs, fmt='o', markersize = 5, color='black',
             ecolor='grey', elinewidth=3, capsize=5);
    plt.title("95% confidence intervals of " + title)
    # styling
    plt.grid(True)
    plt.tick_params(grid_alpha=0.5)
    # naming the x axis
    plt.xlabel(size_col)
    # naming the y axis
    plt.ylabel('arithmetic mean execution time (ms)')
    
    # setup ConfidenceIntervals directory
    filename2 = 'CI_{}.png'.format(t)
    print("--> Storing results to {}".format(filename2))
    plt.savefig("./{}/{}/{}".format(PLOTS_DIR, CI_DIR, filename2))
    # function to save and show the plot
    plt.close()
    
    # save list of points
    fill_graphs(t, pts)


def plot_execution_time_summary_graphs(size_col: str):
    """ generates execution time summary graphs to facilitate implementation comparisons """

    print("\nGenerating execution time summary plots...")

    for t, points in graphs.items():
        print('HELLO')
        x, y = zip(*points)
        plt.plot(x, y, label=t, marker='o', markerfacecolor='black', markersize=5)

 
    #add x-axis values to plot
    plt.xticks(ticks=x, labels=x)
    # styling
    plt.grid(True)
    plt.tick_params(grid_alpha=0.5)
    _, top_y = plt.ylim()
    margin = 0.1
    plt.ylim(bottom=0, top=(1 + margin)*top_y)

    # naming the x axis
    plt.xlabel(size_col)
    # naming the y axis
    plt.ylabel('mean execution time (ms)')

    # giving a title to my graph
    title = 'Execution times for GPU implementations'
    plt.title(title)
    plt.legend()

    # function to show the plot
    filename = "{}.png".format("summary_execution_times")
    print("--> Storing results to {}/{}".format(PLOTS_DIR, filename))
    plt.savefig(
        "./{}/{}".format(PLOTS_DIR, filename))
    plt.clf()


#####################################
# ******** Helper functions **********
#####################################
def preprocess(df: DataFrame) -> Tuple[int, DataFrame, Tuple[int, int], float]:
    """ uses Tukey's method for outlier detection and computes the mean """

    q1 = df['execution_time'].quantile(0.25)
    q3 = df['execution_time'].quantile(0.75)
    iqr = q3-q1
    inner_fence = 1.5*iqr
    outer_fence = 3*iqr

    # inner fence lower and upper end
    inner_fence_le = q1-inner_fence
    inner_fence_ue = q3+inner_fence

    # outer fence lower and upper end
    outer_fence_le = q1-outer_fence
    outer_fence_ue = q3+outer_fence

    outliers_prob = []  # probable outliers
    outliers_poss = []  # possible outliers

    for _, x in enumerate(df['execution_time']):
        if x <= outer_fence_le or x >= outer_fence_ue:
            outliers_prob.append(x)
    for _, x in enumerate(df['execution_time']):
        if x <= inner_fence_le or x >= inner_fence_ue:
            outliers_poss.append(x)

    # remove the rows in the dataframe containing probable outliers
    for i in outliers_prob:
        df = df[df.execution_time != i]

    print("Removed {} outliers".format(len(outliers_prob)))

    # get mean and scale to milliseconds
    m = df.execution_time.describe().loc['mean'] * 0.001
    
    # calculate 95% confidence interval through bootstrapping
    s = df.execution_time.describe().loc['std'] * 0.001
    dof = len(df)-1
    confidence = 0.95
    t_crit = np.abs(t.ppf((1-confidence)/2,dof))
    diff = s*t_crit/np.sqrt(len(df))
    ci = (m-s*t_crit/np.sqrt(len(df)), m+s*t_crit/np.sqrt(len(df)))
    
    #confidence interval of nonparametric data
    df = df.sort_values(by ='execution_time' )
    n = len(df)
    z = 1.96
    rankLow = math.floor((n-z*np.sqrt(n))/2)
    rankHigh = math.floor(1 + (n+z*np.sqrt(n))/2)
    ci_par = (df['execution_time'].iloc[rankLow]*0.001, df['execution_time'].iloc[rankHigh-1]*0.001)

    return m, df, ci, diff
    


def fill_graphs(t: str, points):
    """ Fills the necessary data into each summary graph
    """
    graphs[t] = points


#####################################
# ******** Main **********
#####################################
if __name__ == "__main__":
    if not ((len(sys.argv) == 4 and sys.argv[1] == 'plot')):
        print("Incorrect number of arguments: ", len(sys.argv), flush=True)
        print("example call: python3 plot path/to/resultzip plots.zip")
        exit(-1)

    if sys.argv[1] == "plot":
        if (sys.argv[2].split('.')[-1] == 'zip'):
            generate_for_zip(sys.argv[2], sys.argv[3])
        else:
            generate_statistics_by_type(sys.argv[2], sys.argv[3])
    else:
        print("expected command to be 'plot' got ",
              sys.argv[1], flush=True)
        exit(-1)

#generate_statistics_by_type('/Users/saahitiprayaga/dphpc_project/lu/lu_gpu_benchmarking', 'results.zip')
        
 #p = '/Users/saahitiprayaga/dphpc_project/lu-euler-large-bench-results'
# generate_statistics_by_type(p) /Users/saahitiprayaga/dphpc_project/
#generate_statistics_by_type('/Users/saahitiprayaga/dphpc_project/result-zips/gpu_benchmark_deriche', 'figures_gpu_benchmark_deriche.zip')


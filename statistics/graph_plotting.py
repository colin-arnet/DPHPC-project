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
import seaborn as sns
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

    df_multi_node = []
    df_single_node = []

    def impl_key_func(f): return f.name.split('-')[0]
    result_files = list(os.scandir(path))
    result_files.sort(key=impl_key_func)

    for impl_type, file_iter in itertools.groupby(
            result_files, impl_key_func):
        if re.fullmatch('.*_r[0-9]{1,2}', impl_type):
            if 'r0' in impl_type:
                print("Adding files of type {} to multi-node list".format(impl_type))
                df_multi_node.append(pd.concat((pd.read_csv(file, skiprows=4, sep='\s+', engine='python')
                                                for file in file_iter), ignore_index=True))
        else:
            print("Adding files of type {} to single-node list".format(impl_type))
            df_single_node.append(pd.concat((pd.read_csv(file, skiprows=4, sep='\s+', engine='python')
                                             for file in file_iter), ignore_index=True))

    for df in df_multi_node:
        # split into problem size
        if 'matrix_size' in df.columns:
            size_col = 'matrix_size'
        else:
            size_col = 'w'

        grouped = df.groupby(df[size_col])
        for size in df[size_col].unique():
            df_size = grouped.get_group(size)
            # create process/mean execution time graphs for each size and problem
            # The function also fills the necessary datapoints into the 7 summary graphs
            plot_by_size(df_size, size, True)

    for df in df_single_node:
        # split into problem size
        if 'matrix_size' in df.columns:
            size_col = 'matrix_size'
        else:
            size_col = 'w'

        grouped = df.groupby(df[size_col])
        for size in df[size_col].unique():
            df_size = grouped.get_group(size)
            # create process/mean execution time graphs for each size and problem
            # The function also fills the necessary datapoints into the 7 summary graphs
            plot_by_size(df_size, size, False)

    # plot the summary graphs, result is saved in /dphpc_project/Plots/SummaryPlots
    plot_execution_time_summary_graphs()
    plot_speedup_summary_graphs()

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

def plot_by_size(df: DataFrame, size: int, multi_node: bool):
    """ This function plots process-mean execution time graphs for each implementation
        Datapoints are stored into the summary graphs, which are needed later
    """
    variable = ""
    # type of current df
    t = df['type'].iloc[0]

    print("\nGenerating {} plots for size {}...".format(t, size))

    if multi_node:
        variable = "num_ranks"
    else:
        variable = "num_threads"

    # list of coordinate points (x,y) where x = numThreads/numProcesses and y = mean execution time
    points = []
    # split into thread/rank size
    grouped_df = df.groupby(variable)
    for threads in df[variable].unique():
        df2 = grouped_df.get_group(threads)
        print("----- Processing for {} threads -----".format(threads))
        m, df2, ci, diff = preprocess(df2)
        if multi_node:
            threads = threads * 4  # number of processes
        print("Mean execution time: {}".format(m))
        print("confidence interval: {}".format(ci))
        points.append(((threads, m), diff))

    # split points into x and y coordinates
    points.sort(key=lambda p: p[0][0])
    pts, diffs = zip(*points)
    x, y = zip(*pts)
    
    plt.plot(x, y, color='green', linestyle='dashed', linewidth=3,
             marker='o', markerfacecolor='blue', markersize=10)

    
    # plotting the points
    if multi_node:
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
    if multi_node:
        plt.xlabel('processes')
    else:
        plt.xlabel('threads')

    # naming the y axis
    plt.ylabel('arithmetic mean execution time (ms)')

    # giving a title to my graph
    title = t + ' with matrix size ' + str(size)
    plt.title(title)
    
    '''
    #add coordinates
    for xs,ys in zip(x,y):

        label = "{:.2f}".format(ys)

        plt.annotate(label, # this is the text
                 (xs,ys), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='left') # horizontal alignment can be left, right or center
    '''

    filename = '{}/{}_size_{}.png'.format(PLOTS_DIR, t, size)
    print("--> Storing results to {}".format(filename))

    # function to save and show the plot
    plt.savefig("./{}".format(filename))
    plt.clf()

    # save list of points in graph1 or graph2
    fill_graphs(size, t, pts, multi_node)

    # generate CI plots
    print("\nGenerating confidence interval plots...")

    diffs = np.array(diffs)

    plt.errorbar(x, y, yerr=diffs, fmt='o', markersize=5, color='black',
                 ecolor='grey', elinewidth=3, capsize=5)
    plt.title("95% confidence intervals of " + title)
    # styling
    plt.grid(True)
    plt.tick_params(grid_alpha=0.5)
    top_margin = 0
    for idx, i in enumerate(y):
        if (y[idx] + diffs[idx] > top_margin):
            top_margin = y[idx]+diffs[idx]
    plt.ylim(bottom=0, top=top_margin * 1.1)
    # naming the x axis
    if multi_node:
        plt.xlabel('processes')
    else:
        plt.xlabel('threads')
    # naming the y axis
    plt.ylabel('arithmetic mean execution time (ms)')

    # setup ConfidenceIntervals directory
    filename2 = 'CI_{}_size_{}.png'.format(t, size)
    print("--> Storing results to {}".format(filename2))
    plt.savefig("./{}/{}/{}".format(PLOTS_DIR, CI_DIR, filename2))
    # function to save and show the plot
    plt.clf()


def plot_speedup_summary_graphs():
    """ generates speedup summary graphs to facilitate implementation comparisons """

    # setup SummaryPlots directory
    os.mkdir(os.path.join(PLOTS_DIR, SPEEDUP_DIR))

    print("\nGenerating speedup summary plots...")

    for size, experiments in graphs.items():

        base_time = None
        t_base_time = None
        num_t_base_time = None
        if 'single-node' in experiments.keys():
            measurements = experiments['single-node']
            for t, points in measurements:
                if 'sequential' in t.lower() and points:
                    # speedup relative to sequential execution time
                    base_time = points[0][1]
                    t_base_time = t.lower()
                    num_t_base_time = 1

        for node_setup, measurements in experiments.items():
            if measurements:

                print(
                    "----- Processing for size {} in {} setup -----".format(size, node_setup))

                if not base_time:
                    # Base time is slowest implementation on 1 (or 4 on multi-node) thread
                    min_threads = 2 if node_setup == 'single-node' else 4
                    base_time = 0
                    for t, points in measurements:
                        for x, y in points:
                            if x == min_threads and y > base_time:
                                base_time = y
                                t_base_time = t.lower()
                                num_t_base_time = x

                for t, points in measurements:
                    if (t == 'DAG'):
                        t = 'Kokkos TS'
                    x, y = zip(*points)
                    plt.plot(
                        x, tuple(map(lambda y_val: base_time/y_val, y)), label=t, marker='o', markerfacecolor='black', markersize=5)
                    
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
                plt.ylim(bottom=-1, top=(1 + margin)*top_y)

                # naming the x axis
                plt.xlabel('Threads')
                # naming the y axis
                plt.ylabel('Speedup')

                # giving a title to my graph
                # title = 'Speedups of {} implemenations with matrix size {} relative to {} ms'.format(
                #     node_setup, size, '%.3f' % base_time)
                title = 'Speedups of {} implemenations with matrix size {} relative to {} on {} {}'.format(
                    node_setup, size, t_base_time, num_t_base_time, 'ranks' if 'multi-node' in node_setup else 'threads')
                plt.title(title, wrap=True)
                plt.legend()

                # function to show the plot
                filename = "{}_size_{}.png".format(node_setup, size)
                print("--> Storing results to {}/{}/{}".format(PLOTS_DIR,
                      EXEC_TIMES_DIR, filename))
                plt.savefig(
                    "./{}/{}/{}".format(PLOTS_DIR, SPEEDUP_DIR, filename))
                

                plt.clf()


def plot_execution_time_summary_graphs():
    """ generates execution time summary graphs to facilitate implementation comparisons """

    # setup SummaryPlots directory
    os.mkdir(os.path.join(PLOTS_DIR, EXEC_TIMES_DIR))

    print("\nGenerating execution time summary plots...")

    for size, experiments in graphs.items():
        for node_setup, measurements in experiments.items():
            if measurements:

                print(
                    "----- Processing for size {} in {} setup -----".format(size, node_setup))

                for t, points in measurements:
                    x, y = zip(*points)
                    plt.plot(x, y, label=t, marker='o', markerfacecolor='black', markersize=5)
                    '''
                    for i,j in zip(x,y):
                        label = "{:.2f}".format(ys)

                        plt.annotate(label, # this is the text
                         (xs,ys), # these are the coordinates to position the label
                         textcoords="offset points", # how to position the text
                         xytext=(0,10), # distance from text to points (x,y)
                         ha='left') # horizontal alignment can be left, right or center
                    '''



        
                
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
                plt.xlabel('threads')
                # naming the y axis
                plt.ylabel('mean execution time (ms)')

                # giving a title to my graph
                title = 'Execution times of {} implemenations with matrix size {}'.format(
                    node_setup, size)
                plt.title(title)
                plt.legend()
                
        
                # function to show the plot
                filename = "{}_size_{}.png".format(node_setup, size)
                print("--> Storing results to {}/{}/{}".format(PLOTS_DIR,
                      EXEC_TIMES_DIR, filename))
                plt.savefig(
                    "./{}/{}/{}".format(PLOTS_DIR, EXEC_TIMES_DIR, filename), bbox_inches = 'tight')
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
    t_crit = np.abs(t.ppf((1-confidence)/2, dof))
    diff = s*t_crit/np.sqrt(len(df))
    ci = (m-s*t_crit/np.sqrt(len(df)), m+s*t_crit/np.sqrt(len(df)))

    # confidence interval of nonparametric data
    df = df.sort_values(by='execution_time')
    n = len(df)
    z = 1.96
    rankLow = math.floor((n-z*np.sqrt(n))/2)
    rankHigh = math.floor(1 + (n+z*np.sqrt(n))/2)
    if not 'sequential' in df['type'].iloc[0].lower():
        ci_par = (df['execution_time'].iloc[rankLow]*0.001,
                  df['execution_time'].iloc[rankHigh-1]*0.001)

    return m, df, ci, diff


def fill_graphs(size: int, t: str, points, multi_node: bool = True):
    """ Fills the necessary data into each summary graph. I.e, graphs 1,2,3,4 contain the datapoints for
        MPI_Kokkos vs. MPI_OMP with their respective input sizes, graphs 5,6,7 for OMPvs.Kokkosvs.DAG.
    """
    if not size in graphs:
        graphs[size] = {"multi-node": [], "single-node": []}
    if multi_node:
        graphs[size]["multi-node"].append((t, points))
    else:
        graphs[size]["single-node"].append((t, points))


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



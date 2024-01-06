import pandas as pd
import numpy as np 
import math
from scipy.stats import t
import scipy.stats as st
import os
import glob
from zipfile import ZipFile


def generate_statistics(resultfile):
    df = pd.read_csv(resultfile, skiprows = 4, sep='\s+', engine='python')
    #split into problem size
    grouped = df.groupby(df['matrix_size'])
    for size in df['matrix_size'].unique():
        temporary_df = grouped.get_group(size)
        #remove outliers
        m, temporary_df = preprocess(temporary_df)
        generate_stats_by_problem_size(resultfile, temporary_df, size)
    
##########################################################################################
#********  Summary statistics  **********
#Outputs for each file in /dphpc_project/lu-euler-large-bench-results
#summary statistics consisting of arithmetic mean, variance, standard deviation, confidence95
#(which can later be removed, it is the 95% confidence interval that is calculated from the 
#given runs), confidenceN (the actual confidence interval we care about, the one that we defined
#according to the paper as [m-em, m+em], where 1-e is the acceptable error of 10%) (assuming normalverteilt!)
#, and the n (number of needed measurements) we can calculate from that.
#########################################################
def generate_stats_by_problem_size(resultfile, df, size):
    df2 = df.execution_time.describe().loc[['mean', 'std']]
    df2.loc['variance'] = df2.loc['std']**2
    
    #calculate 95% confidence interval through bootstrapping algorithm
    m = df.execution_time.describe().loc['mean']
    s = df.execution_time.describe().loc['std']
    dof = len(df)-1 
    confidence = 0.95
    t_crit = np.abs(t.ppf((1-confidence)/2,dof))
    range1 = (m-s*t_crit/np.sqrt(len(df)), m+s*t_crit/np.sqrt(len(df))) 
    df2.loc['confidence95'] = range1
    
    #number of measurements
    #acceptable error defined as 0.1
    confN = (m - 0.9*m, m + 0.9*m)
    df2.loc['confidenceN'] = confN
    n = ((s*t_crit)/(0.9*m)) ** 2
    if (n < 1):
        n = 0
    else :
        n = math.ceil(n)
    df2.loc['current runs'] = len(df)
    df2.loc['additionally needed runs'] = n
    #print("\n Summary statistics (in ms) for " + os.path.basename(resultfile) + " with matrix size ", df['matrix_size'].iloc[0])
    print("\n Summary statistics (in ms) for " + resultfile + " with matrix size ", df['matrix_size'].iloc[0])
    print(df2)

#####################################
#******** Helper function **********
#####################################
def preprocess(df):
    #uses Tukey's method for outlier detection
    #Takes two parameters: dataframe & variable of interest as string
    q1 = df['execution_time'].quantile(0.25)
    q3 = df['execution_time'].quantile(0.75)
    iqr = q3-q1
    inner_fence = 1.5*iqr
    outer_fence = 3*iqr
    
    #inner fence lower and upper end
    inner_fence_le = q1-inner_fence
    inner_fence_ue = q3+inner_fence
    
    #outer fence lower and upper end
    outer_fence_le = q1-outer_fence
    outer_fence_ue = q3+outer_fence
    
    outliers_prob = [] #probable outliers
    outliers_poss = [] #possible outliers
    for index, x in enumerate(df['execution_time']):
        if x <= outer_fence_le or x >= outer_fence_ue:
            outliers_prob.append(x)
    for index, x in enumerate(df['execution_time']):
        if x <= inner_fence_le or x >= inner_fence_ue:
            outliers_poss.append(x)
    #remove the rows in the dataframe containing probable outliers
    for i in outliers_prob:
        df = df[df.execution_time != i]
    m = df.execution_time.describe().loc['mean']
    return m, df  
 
    
    
    
    
zippath = '/Users/saahitiprayaga/dphpc_project/lu-euler-large-bench-results.zip'
# Create a ZipFile Object and load results.zip in it
with ZipFile(zippath, 'r') as zipObj:
    dir_name = "tmp_result_dir"
    zipObj.extractall(dir_name)
    result_zip = 'summaryStatistics.zip'
    for resultfile in os.scandir(dir_name):
            #ignore the files where rank of MPI is not 0
            if (resultfile.path.startswith('tmp_result_dir/lu_mpi_omp_r')) or (resultfile.path.startswith('tmp_result_dir/lu_mpi_kokkos_r')):
                if not ((resultfile.path.startswith('tmp_result_dir/lu_mpi_omp_r0')) or (resultfile.path.startswith('tmp_result_dir/lu_mpi_kokkos_r0'))):
                        continue
            generate_statistics(resultfile.path)
       
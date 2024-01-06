import re
import sys
import zipfile
import os
from shutil import rmtree
from typing import *

import numpy as np
import pandas
import statsmodels.api as sm


def generate_qq(resultfile: str, qq_plot_dir='.'):
    """ generate a qq-plot for each tuple of parameters in the resultfile """
    print("Reading data from results file {}".format(resultfile.split('/')))
    r = pandas.read_csv(resultfile, header=4, sep='\s+', engine='python')
    curr_size = (0, 0)
    curr_measurements = []
    for _, row in r.iterrows():
        if 'w' in r.columns:
            if (row['w'], row['h']) == curr_size:
                curr_measurements.append(float(row['execution_time']))
            else:
                output_qq_plot(curr_measurements, resultfile,
                               curr_size, qq_plot_dir)
                curr_size = (row['w'], row['h'])
                curr_measurements = [float(row['execution_time'])]
        else:
            m_size = row['matrix_size']
            b_size = row['block_size'] if 'block_size' in r.columns else 0
            if (m_size, b_size) == curr_size:
                curr_measurements.append(float(row['execution_time']))
            else:
                output_qq_plot(curr_measurements, resultfile,
                               curr_size, qq_plot_dir)
                curr_size = (m_size, b_size)
                curr_measurements = [float(row['execution_time'])]
    output_qq_plot(curr_measurements, resultfile, curr_size, qq_plot_dir)


def output_qq_plot(measurements: List[float], resultfile: str, size: Tuple[int, int], qq_plot_dir='.'):
    if measurements:
        print(
            "Generating qq-plot for size ({}, {})...".format(size[0], size[1]))
        fig = sm.qqplot(np.asfarray(measurements), fit=True, line='45')
        filename = os.path.join(qq_plot_dir, "{}_qq_{}_{}.png".format(
            resultfile.split('/')[-1].split('.')[0], size[0], size[1]))
        fig.savefig(filename)
        print("Output qq-plot to", filename)


def qq_plot_for_zip(zippath: str):
    """ Generates a zip containing qq plots for each file in resultzip """
    with zipfile.ZipFile(zippath, 'r') as zip_ref:
        dir_name = "tmp_result_dir"
        zip_ref.extractall(dir_name)
        result_zip = 'qq-plots.zip'
        f_result_zip = zipfile.ZipFile(result_zip, 'w', zipfile.ZIP_DEFLATED)
        print("{} contains {}".format(dir_name, os.listdir(dir_name)))
        for resultfile in os.scandir(dir_name):
            generate_qq(resultfile.path, dir_name)
        for file in os.scandir(dir_name):
            if re.fullmatch('.*_qq_.*\.png', file.name):
                f_result_zip.write(file.path, file.name)
        f_result_zip.close()
        rmtree(dir_name)


if __name__ == "__main__":
    if not ((len(sys.argv) == 3 and sys.argv[1] == 'qq-plot')):
        print("Incorrect number of arguments: ", len(sys.argv), flush=True)
        exit(-1)

    if sys.argv[1] == "qq-plot":
        if (sys.argv[2].split('.')[-1] == 'zip'):
            qq_plot_for_zip(sys.argv[2])
        else:
            generate_qq(sys.argv[2])
    else:
        print("expected command to be 'qq-plot' got ",
              sys.argv[1], flush=True)
        exit(-1)

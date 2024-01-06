import json
import sys
import os
import shutil
import zipfile
import re
import subprocess

# Runtime build variables: base build dir (contains config.json), kokkos dir
# Runtime run variables: base build dir (contains config.json), matrix/block size config, no of runs

CONFIG_NAME = "compile_run_config.json"

# set placeholder to projname
RESULTS_NAME = "{0}(_r[0-9]+)?-result-[0-9]+.txt"


def compile_all(base_build_dir: str, kokkos_dir: str):
    """ build all experiments """
    f_config = open(os.path.join(base_build_dir, CONFIG_NAME))
    config_data = json.load(f_config)
    for config in config_data['build-configs']:
        build_dir = os.path.join(base_build_dir, config['build-dir'])
        # resolve placeholders
        build_cmd = config['build-cmd'].format(kokkos_dir)
        print("======================== {} ========================".format(
            build_dir), flush=True)
        if subprocess.call("mkdir {}; cd {}; {}".format(build_dir, build_dir, build_cmd), shell=True) == -1:
            exit(-1)
        print("", flush=True)


def run_all_single_node(base_build_dir: str):
    """ run all single-node type experiments """
    f_config = open(os.path.join(base_build_dir, CONFIG_NAME))
    config_data = json.load(f_config)
    size_configs = ' '.join([' '.join([str(config['w']), str(config['h'])])
                             for config in config_data['size-configs']])
    for run_config in config_data['run-configs-single-node']:
        build_dir = os.path.join(base_build_dir, run_config['build-dir'])
        num_runs = run_config["num-runs"]
        print(
            "======================== {} ========================".format(build_dir))
        for num_threads in config_data['bench-configs-single-node']:
            print("Setting number of threads to {}".format(
                num_threads), flush=True)
            # resolve placeholders
            run_cmd = run_config['run-cmd'].format(num_threads,
                                                   num_runs, size_configs)
            subprocess.call("cd {}; {}".format(build_dir, run_cmd), shell=True)
            print("")


def run_all_multi_node(base_build_dir: str):
    """ run all multi-node type experiments """
    f_config = open(os.path.join(base_build_dir, CONFIG_NAME))
    config_data = json.load(f_config)
    size_configs = ' '.join([' '.join([str(config['w']), str(config['h'])])
                             for config in config_data['size-configs']])
    for run_config in config_data['run-configs-multi-node']:
        build_dir = os.path.join(base_build_dir, run_config['build-dir'])
        num_runs = run_config["num-runs"]
        print(
            "======================== {} ========================".format(build_dir), flush=True)
        for bench_config in config_data['bench-configs-multi-node']:
            num_nodes = bench_config['num-nodes']
            num_threads = bench_config['num-threads']
            print("Setting number of nodes to {} and threads to {}".format(
                num_nodes, num_threads), flush=True)
            # resolve placeholders
            run_cmd = run_config['run-cmd'].format(num_nodes,
                                                   num_threads, num_runs, size_configs)
            subprocess.call("cd {}; {}".format(build_dir, run_cmd), shell=True)
            print("", flush=True)


def gather_results(base_build_dir: str):
    """ copy result files into single folder and zip """
    f_config = open(os.path.join(base_build_dir, CONFIG_NAME))
    config_data = json.load(f_config)
    result_zip = os.path.join(
        base_build_dir, 'results.zip')
    f_result_zip = zipfile.ZipFile(result_zip, 'w', zipfile.ZIP_DEFLATED)
    for config in config_data['build-configs']:
        build_dir = os.path.join(base_build_dir, config['build-dir'])
        results_regex = RESULTS_NAME.format(config['build-dir'].split('/')[0])
        for file in os.listdir(build_dir):
            if re.fullmatch(results_regex, file):
                f_result_zip.write(os.path.join(build_dir, file), file)
    f_result_zip.close()
    print("Gathered results in file ", result_zip, flush=True)


def clean_files(base_build_dir: str):
    """ iterate through build dirs, and clean build and result files """
    f_config = open(os.path.join(base_build_dir, CONFIG_NAME))
    config_data = json.load(f_config)
    for config in config_data['build-configs']:
        build_dir = os.path.join(base_build_dir, config['build-dir'])
        # resolve placeholders
        print("Build dir: ", build_dir, flush=True)
        if subprocess.call("cd {}; make clean".format(build_dir), shell=True) == -1:
            exit(-1)

        build_p_segments = config['build-dir'].split('/')
        if len(build_p_segments) > 1 and build_p_segments[1] == 'build':
            # remove all contents of build directory, ESPECIALLY cmake cache
            results_regex = '.*'
        else:
            # only remove result files
            results_regex = RESULTS_NAME.format(build_p_segments[0])

        for file in os.scandir(build_dir):
            if re.fullmatch(results_regex, file.name):
                print("Removing ", file.name, flush=True)
                if file.is_dir():
                    shutil.rmtree(file.path)
                elif file.is_file():
                    os.remove(file.path)


if __name__ == "__main__":
    if not ((len(sys.argv) == 4 and sys.argv[1] == 'build') or (len(sys.argv) == 3 and sys.argv[1] == 'run') or (len(sys.argv) == 3 and sys.argv[1] == 'clean')):
        print("Incorrect number of arguments: ", len(sys.argv), flush=True)
        exit(-1)

    if sys.argv[1] == "build":
        compile_all(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "run":
        run_all_single_node(sys.argv[2])
        run_all_multi_node(sys.argv[2])
        gather_results(sys.argv[2])
    elif sys.argv[1] == "clean":
        clean_files(sys.argv[2])
    else:
        print("expected command to be either 'build' or 'run', got ",
              sys.argv[1], flush=True)
        exit(-1)

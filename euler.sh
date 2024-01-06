#!/bin/bash

set -e

# see https://scicomp.ethz.ch/wiki/Getting_started_with_clusters

HOSTALIAS="euler"

PYTHONFILE="compile_run.py"
TASKZIP="task.zip"
RESULTZIP="results.zip"
MAKEFILE="Makefile"
EXEC="exec"

GIT_VERSION="2.11.0" # one of 1.9.0, 2.11.0
GCC_VERSION="6.3.0" # one of 4.7.4, 4.4.7‌, 4.8.2‌, 4.9.2, 4.8.4‌, 5.2.0‌, 6.2.0‌, 6.3.0
CMAKE_VERSION="3.13.5" # one of 2.8.12‌, 3.3.1‌, 3.5.2, 3.9.2
OMP_VERSION="3.0.0" # one of 1.4.5, 1.6.5, 1.10.0‌, 1.10.2‌, 1.10.3‌, 1.8.0‌, 2.0.0‌, 2.0.2‌, 2.1.1‌, 2.1.2‌, 3.0.0
PYTHON_VERSION="3.6.1" # one of 2.7.12‌ 2.7.13‌ 2.7.6‌ 2.7.9‌ 3.3.3‌ 3.4.3‌ 3.6.0 2.7.14‌2.7.6_UCS4‌3.6.1

MODULES="new git/$GIT_VERSION eth_proxy gcc/$GCC_VERSION cmake/$CMAKE_VERSION open_mpi/$OMP_VERSION python/$PYTHON_VERSION"

function usage() {
    echo "Script to submit C(++) batch jobs to the euler cluster"
    echo ""
    echo "The cluster is only accessible from inside the ETH network."
    echo "If you would like to connect from a computer, which is not inside"
    echo "the ETH network, then you would need to establish a VPN connection first."
    echo "Add your ssh keys to the cluster to login without having to type a password."
    echo "See https://scicomp.ethz.ch/wiki/Getting_started_with_clusters#SSH_keys"
    echo "Add the generated ssh key to your ssh config under the alias '$HOSTALIAS'"
    echo ""
    echo "Usage:"
    echo "$0 help"
    echo "      show this help"
    echo ""
    echo "$0 submit <USERNAME> <NUM_CORES> <PROCS_PER_NODE> <MEM_PER_CORE> <RUN_TIME> <FOLDER>"
    echo "      Submit a batch job with the given parameters."
    echo "      For this <FOLDER> needs to contain a python file $PYTHONFILE which offers "
    echo "      build, run and clean commands"
    echo "      The result zip, if there is any, has to be called '$RESULTZIP'"
    echo ""
    echo "$0 list <USERNAME> "
    echo "      List currently running batch jobs"
    echo ""
    echo "$0 view <USERNAME> <JOBID>"
    echo "      View details for specific job"
    echo ""
    echo "$0 kill <USERNAME> <JOBID>"
    echo "      Kill a specific job"
    echo ""
    echo "$0 rlogs <USERNAME> <JOBID>"
    echo "      Inspect logs of a running job"
    echo ""
    echo "$0 flogs <USERNAME> <JOBID>"
    echo "      Inspect logs of a finished job"
    echo ""
    echo "$0 result <USERNAME> <FOLDER>"
    echo "      download the result zip, if there is any, from the last finished job"
    echo ""
    echo "Examples: "
    echo "$0 submit flbuetle 48 8 256 06:00 lu/"
    echo "$0 list flbuetle"
    echo "$0 view flbuetle 118294868"
    echo "$0 kill flbuetle 118294868"
    echo "$0 rlogs flbuetle 118294868"
    echo "$0 flogs flbuetle 118294868"
    echo "$0 result flbuetle"
}

function is_username_present() {
    USERNAME=$1
    if [ -z "$USERNAME" ]; then
        echo "Not all required parameters where provided: username"
        exit 1
    fi
}

function generic_query() {
    USERNAME=$1
    is_username_present $USERNAME
    ssh -T $USERNAME@$HOSTALIAS $2
}

function submit() {
    USERNAME=$1
    NUM_CORES=$2
    PROCS_PER_NODE=$3
    MEM_PER_CORE=$4
    RUN_TIME=$5
    FOLDER=$6

    is_username_present $USERNAME

    WORKSPACE="/cluster/home/$USERNAME/script-workspace"
    KOKKOS_DIR="$WORKSPACE/kokkos"

    if [ -z "$NUM_CORES" ] || [ -z "$PROCS_PER_NODE" ] || [ -z "$RUN_TIME" ] || [ -z "$MEM_PER_CORE" ] || [ -z "$FOLDER" ]; then
        echo "Not all required parameters where provided"
        exit 1
    fi

    if  [ ! -d $FOLDER ]; then
            echo -e "$1 not found in current working directory, exiting ..."
            exit 1
    fi

    if [ -f $TASKZIP ]; then
            echo -e "Found old task zip locally, deleting it ..."
            rm $TASKZIP
    fi

    if [ -f $FOLDER/$RESULTZIP ]; then
            echo -e "Found old result zip locally, deleting it ..."
            rm $FOLDER/$RESULTZIP
    fi

    echo "Zipping files..."
    zip -q -r $TASKZIP $FOLDER

    ssh -T $USERNAME@$HOSTALIAS <<ENDSSH
if [ -d $WORKSPACE ]; then
    echo -e "Found old script workspace remotely, cleaning up ..."
    rm -r $WORKSPACE/*
else
    mkdir $WORKSPACE
fi
ENDSSH

    echo "Tranferring files ..."
    scp $TASKZIP $USERNAME@$HOSTALIAS:$WORKSPACE

    echo "Deleting task zip locally ..."
    rm $TASKZIP

    echo "Preparing environment ..."
    ssh -T $USERNAME@$HOSTALIAS <<ENDSSH
module load $MODULES
cd $WORKSPACE
git clone https://github.com/kokkos/kokkos.git
git -C ./kokkos/ checkout 4d23839
unzip -q -o $TASKZIP
cd $FOLDER
python3 $PYTHONFILE clean .
python3 $PYTHONFILE build . $KOKKOS_DIR
ENDSSH

    echo "Submitting batch job ..."
    ssh $USERNAME@$HOSTALIAS bsub -n $NUM_CORES -W $RUN_TIME -R "rusage[mem=$MEM_PER_CORE]" -R "span[ptile=$PROCS_PER_NODE]" <<ENDBSUB
module load $MODULES
cd $WORKSPACE
cd $FOLDER
python3 $PYTHONFILE run .
ENDBSUB
}

function result() {
    USERNAME=$1
    FOLDER=$2

    is_username_present $USERNAME

    WORKSPACE="/cluster/home/$USERNAME/script-workspace"

    if ! ssh -T $USERNAME@$HOSTALIAS "ls $WORKSPACE/$FOLDER"
    then
        echo -e "Remote workspace not found. You need to submit a job first, exiting ..."
        exit 1
    fi

    scp $USERNAME@$HOSTALIAS:$WORKSPACE/$FOLDER/$RESULTZIP ./
}

cd "$(dirname "$0")"

case "$1" in
    help)
        usage
        exit 0
        ;;
    submit)
        submit $2 $3 $4 $5 $6 $7
        exit 0
        ;;
    list)
        generic_query $2 "bjobs"
        exit 0
        ;;
    view)
        generic_query $2 "bbjobs $3"
        exit 0
        ;;
    kill)
        generic_query $2 "bkill $3"
        exit 0
        ;;
    rlogs)
        generic_query $2 "bpeek $3"
        exit 0
        ;;
    flogs)
        generic_query $2 "cat lsf.o$3"
        exit 0
        ;;
    result)
        result $2 $3
        exit 0
        ;;
    *)
        usage
        exit 1
esac

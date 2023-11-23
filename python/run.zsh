#!/bin/zsh
source ../parallel-venv/bin/activate
yaml() {
    python3 -c "import yaml;print(yaml.safe_load(open('$1'))$2)"
}

VALUE=$(yaml ./config.yaml "['number_of_processes']")
echo Number of processes: $VALUE

if [[ $VALUE -le 0 ]]; then
    echo "Number of processes must be greater than 0"
    exit 0
fi

if [[ $VALUE -eq 1 ]]
then
    echo "Running serial code"
    python3 ./compute.py
    exit 0
else
    echo "Running with MPI"
    mpiexec -n $VALUE python3 ./compute_mpi.py
    exit 0
fi



#!/bin/bash
echo "PROFILING USING cProfile:"

## Set custom venv
source parallel-venv/bin/activate

## Function that read config file
yaml() {
    python3 -c "import yaml;print(yaml.safe_load(open('$1'))$2)"
}

## Get number of processes/tasks
config_file='python/config.yaml'
NUMBER_OF_PROCESSES=$(yaml ${config_file} "['number_of_processes']")
if [[ $NUMBER_OF_PROCESSES -le 0 ]]; then
    echo "Number of processes must be greater than 0"
    exit 0
fi

NUMBER_OF_NODES=$(yaml ${config_file} "['number_of_nodes']")
NUMBER_OF_CPUS_PER_TASK=$(yaml ${config_file} "['number_of_cpus_per_task']")
RUNTIME=$(yaml ${config_file} "['runtime']")
echo Number of processes: $NUMBER_OF_PROCESSES
echo Number of nodes: $NUMBER_OF_NODES
echo Number of cpus per task: $NUMBER_OF_CPUS_PER_TASK
TAG=$(yaml ${config_file} "['run_tag']")
NX=$(yaml ${config_file} "['Nx']")

TAG_str=''
if [ "$TAG" != '' ]; then
    TAG_str="_${TAG}"
    echo Tag: $TAG
fi
TAG_str+="_profiling"

## Slurm commands
echo "#!/bin/bash" > tsunami.job
echo "#SBATCH --account phys-743" >> tsunami.job
echo "#SBATCH --reservation phys-743" >> tsunami.job
echo "#SBATCH --job-name=tsunami${TAG_str}" >> tsunami.job
echo "#SBATCH --output=results/output/tsunami${TAG_str}_ntasks${NUMBER_OF_PROCESSES}_nodes${NUMBER_OF_NODES}_ncpt${NUMBER_OF_CPUS_PER_TASK}_nx${NX}.out" >> tsunami.job
echo "#SBATCH --error=results/output/tsunami${TAG_str}_ntasks${NUMBER_OF_PROCESSES}_nodes${NUMBER_OF_NODES}_ncpt${NUMBER_OF_CPUS_PER_TASK}_nx${NX}.err" >> tsunami.job
echo "#SBATCH --time $RUNTIME" >> tsunami.job
echo "#SBATCH --cpus-per-task $NUMBER_OF_CPUS_PER_TASK" >> tsunami.job
echo "#SBATCH --ntasks $NUMBER_OF_PROCESSES" >> tsunami.job
echo "#SBATCH --nodes $NUMBER_OF_NODES" >> tsunami.job
if [[ $NX == 8001 ]]; then
    echo "#SBATCH --mem-per-cpu 8000" >> tsunami.job
fi

PROFILE_PATH='results/profiling/'
PROFILE_FIG='results/profiling/'

if [[ $NUMBER_OF_PROCESSES -eq 1 ]]
then
    echo "Running serial code"
    PROFILE_PATH+='single'
    PROFILE_PATH+=$TAG_str
    PROFILE_PATH+='.stats'
    PROFILE_FIG+='single'
    PROFILE_FIG+=$TAG_str
    PROFILE_FIG+='.png'
    echo "source parallel-venv/bin/activate" >> tsunami.job
    echo "srun python3 -m cProfile -o ${PROFILE_PATH} python/compute_profiling.py" >> tsunami.job
else
    echo "Running with MPI"
    if [[ $NUMBER_OF_NODES -ge 2 ]]; then
        echo "#SBATCH --qos=parallel" >> tsunami.job
    fi
    PROFILE_PATH+='mpi'
    PROFILE_PATH+=$TAG_str
    PROFILE_PATH+='.stats'
    PROFILE_FIG+='mpi'
    PROFILE_FIG+=$TAG_str
    PROFILE_FIG+='.png'
    echo "source parallel-venv/bin/activate" >> tsunami.job
    echo "srun python3 -m cProfile -o ${PROFILE_PATH} python/compute_mpi_profiling.py" >> tsunami.job
fi

## Produce plot
echo "gprof2dot -f pstats ${PROFILE_PATH}| dot -Tpng -o ${PROFILE_FIG}">> tsunami.job

echo "Path to profile stats: ${PROFILE_PATH}"

sbatch tsunami.job


#!/bin/bash
# SLURM submission script for submitting multiple serial jobs on Niagara
#
#SBATCH --account=def-amw8
#SBATCH --time=02:58:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80
#SBATCH --job-name FirstChainExp


alpha=(__ALPHA__)
run=({__RUN__})
algorithm=(__ALGORITHM__)
environment=(__ENVIRONMENT__)
feature_kind=(__FEATUREKIND__)
problem=(__PROBLEM__)
save_path=(__SAVEPATH__)

module load NiaEnv/2019b
module load gnu-parallel
module load python

cd $SLURM_SUBMIT_DIR || exit
export OMP_NUM_THREADS=1

echo "The number of available cores is echo $NCORES"
echo "Current working directory is $(pwd)"
echo "Running on hostname $(hostname)"
echo "Starting run at: $(date)"

HOSTS=$(scontrol show hostnames $SLURM_NODELIST | tr '\n' ,)
NCORES=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))


parallel --env OMP_NUM_THREADS,PATH,LD_LIBRARY_PATH --joblog slurm-$SLURM_JOBID.log \
-j $NCORES -S $HOSTS --wd $PWD python Learning.py ::: -alg ::: ${algorithm[@]} ::: \
-a ::: ${alpha[@]} ::: -r ::: ${run[@]} ::: -sp ::: ${save_path[@]} ::: \
-e ::: ${environment[@]} ::: -f ::: ${feature_kind[@]} ::: -p ::: ${problem[@]}


echo "Program test finished with exit code $? at: $(date)"
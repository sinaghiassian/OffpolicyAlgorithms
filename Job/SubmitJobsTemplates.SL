#!/bin/bash
# SLURM submission script for submitting multiple serial jobs on Niagara
#
#SBATCH --account=def-sutton
#SBATCH --time=11:58:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name __TASK_____ALGORITHM__


alpha=(__ALPHA__)
lmbda=(__LMBDA__)
eta=(__ETA__)
beta=(__BETA__)
zeta=(__ZETA__)
tdrc_beta=(__TDRCBETA__)
num_of_runs=__NUMOFRUNS__
num_steps=__NUMSTEPS__
sub_sample=__SUBSAMPLE__
algorithm=__ALGORITHM__
environment=__ENVIRONMENT__
task=__TASK__
save_path=__SAVEPATH__

source ~/RLENV/bin/activate
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


parallel --env OMP_NUM_THREADS,PATH,LD_LIBRARY_PATH --joblog slurm-$SLURM_JOBID.log -j $NCORES -S $HOSTS --wd $PWD \
python Learning.py ::: -sp ::: ${save_path} ::: -e ::: ${environment} ::: -alg ::: ${algorithm} ::: -t ::: ${task[@]} \
::: -a ::: ${alpha[@]} ::: -nr ::: ${num_of_runs} ::: -ns ::: ${num_steps} ::: -et ::: ${eta[@]} \
::: -l ::: ${lmbda[@]} ::: -z ::: ${zeta[@]} ::: -tb ::: ${tdrc_beta[@]} ::: -b ::: ${beta[@]} ::: -ss ::: ${sub_sample}


echo "Program test finished with exit code $? at: $(date)"

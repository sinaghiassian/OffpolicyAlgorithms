#!/bin/bash
#SBATCH --account=def-sutton
#SBATCH --time=00:15:58
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --array=1-__NUM_OF_JOBS__

alg=(__ALG__)
source ~/bin/activate/RLENV
$(sed -n "${SLURM_ARRAY_TASK_ID}p" <exports.dat)
echo ${SLURM_ARRAY_TASK_ID} $ALPHA $LMBDA $ETA $BETA $ZETA $NUMOFRUNS $NUMSTEPS $SUBSAMPLE
echo "Current working directory is $(pwd)"
echo "Running on hostname $(hostname)"
echo
echo "Starting run at: $(date)"
python Learning.py \
-a $ALPHA -l $LMBDA -et $ETA -b $BETA -z $ZETA -alg ALGORITHM -t $TASK -nr $NUMOFRUNS -e \
$ENVIRONMENT -sp $SAVE_PATH -ns $NUMSTEPS -ss $SUBSAMPLE
echo "Program test finished with exit code $? at: $(date)"

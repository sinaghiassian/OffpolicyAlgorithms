#!/bin/bash
#SBATCH --account=def-amw8
#SBATCH --time=00:15:58
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --array=1-__NUM_OF_JOBS__
#SBATCH --job-name __NAME_OF_EXP__

alg=__ALG__
source ~/RLENV/bin/activate
`sed -n "${SLURM_ARRAY_TASK_ID}p" <exports_${alg}.dat`
echo ${SLURM_ARRAY_TASK_ID} $ALPHA $LMBDA $ETA $BETA $ZETA $TDRCBETA $NUMOFRUNS $NUMSTEPS $SUBSAMPLE
echo "Current working directory is $(pwd)"
echo "Running on hostname $(hostname)"
echo
echo "Starting run at: $(date)"
python Learning.py \
-a $ALPHA -l $LMBDA -et $ETA -b $BETA -z $ZETA -tb $TDRCBETA -alg $ALGORITHM -t $TASK -nr $NUMOFRUNS -e \
$ENVIRONMENT -sp $SAVE_PATH -ns $NUMSTEPS -ss $SUBSAMPLE
echo "Program test finished with exit code $? at: $(date)"

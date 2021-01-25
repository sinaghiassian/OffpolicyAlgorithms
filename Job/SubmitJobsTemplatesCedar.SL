#!/bin/bash
#SBATCH --account=def-sutton
#SBATCH --time=11:58:59
#SBATCH --cpus-per-task=1
#SBATCH --mem=768mb
#SBATCH --array=1-2

module load scipy-stack/2019a
`sed -n "${SLURM_ARRAY_TASK_ID}p" <exports.dat`
echo ${SLURM_ARRAY_TASK_ID} $runNo $alpha $zeta

echo "Current working directory is `pwd`"
echo "Running on hostname `hostname`"

echo "Starting run at: `date`"
python ../Learning.py ${algName} ${runNo} ${alpha} ${zeta}
echo "Program test finished with exit code $? at: `date`"

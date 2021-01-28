#!/bin/bash
rm -f exports.dat
alpha=(__ALPHA__)
lmbda=(__LMBDA__)
eta=(__ETA__)
beta=(__BETA__)
zeta=(__ZETA__)
num_of_runs=(__NUMOFRUNS__)
num_steps=(__NUMSTEPS__)
sub_sample=(__SUBSAMPLE__)
algorithm=(__ALGORITHM__)
environment=(__ENVIRONMENT__)
task=(__TASK__)
save_path=(__SAVEPATH__)

for A in ${alpha[@]}; do
  for L in ${lmbda[@]}; do
    for E in ${eta[@]}; do
      for B in ${beta[@]}; do
        for Z in ${zeta[@]}; do
          echo export SAVE_PATH=${save_path[0]} ENVIRONMENT=${environment[0]} ALGORITHM=${algorithm[0]} \
          TASK=${task[0]} ALPHA=${A} LMBDA=${L} ETA=${E} BETA=${B} ZETA=${Z} NUMOFRUNS=${num_of_runs[0]} \
          NUMSTEPS=${num_steps[0]} SUBSAMPLE=${sub_sample[0]} >>exports.dat
        done
      done
    done
  done
done

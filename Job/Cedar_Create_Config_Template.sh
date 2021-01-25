#!/bin/bash
rm -f exports.dat
alpha=(__ALPHA__)
lmbda=(__LMBDA__)
eta=(__ETA__)
beta=(__BETA__)
zeta=(__ZETA__)
run=({__RUN__})
algorithm=(__ALGORITHM__)
environment=(__ENVIRONMENT__)
task=(__TASK__)
save_path=(__SAVEPATH__)

for A in ${alpha[@]}; do
  for L in ${lmbda[@]}; do
    for E in ${eta[@]}; do
      for B in ${beta[@]}; do
        for Z in ${zeta[@]}; do
          for R in ${run[@]}; do
            echo export SAVE_PATH=${save_path[*]} ENVIRONMENT=${environment[*]} ALGORITHM=${algorithm[*]} \
            TASK=${task[*]} ALPHA=${A} LMBDA=${L} ETA=${E} BETA=${B} ZETA=${Z} RUN=${R} >>exports.dat
          done
        done
      done
    done
  done
done

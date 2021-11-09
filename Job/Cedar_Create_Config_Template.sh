#!/bin/bash
alpha=(__ALPHA__)
lmbda=(__LMBDA__)
eta=(__ETA__)
beta=(__BETA__)
zeta=(__ZETA__)
tdrc_beta=(__TDRCBETA__)
gem_alpha=(__GEMALPHA__)
gem_beta=(__GEMBETA__)
num_of_runs=__NUMOFRUNS__
num_steps=__NUMSTEPS__
sub_sample=__SUBSAMPLE__
algorithm=__ALGORITHM__
environment=__ENVIRONMENT__
task=__TASK__
save_path=__SAVEPATH__

rm -f exports_${algorithm}.dat
for A in ${alpha[@]}; do
  for L in ${lmbda[@]}; do
    for E in ${eta[@]}; do
      for B in ${beta[@]}; do
        for Z in ${zeta[@]}; do
          for T in ${tdrc_beta[@]}; do
            for GA in ${gem_alpha[@]}; do
              for GB in ${gem_beta[@]}; do
                echo export SAVE_PATH=${save_path} ENVIRONMENT=${environment} ALGORITHM=${algorithm} \
                TASK=${task} ALPHA=${A} LMBDA=${L} ETA=${E} BETA=${B} ZETA=${Z} TDRCBETA=${T} GEMALPHA=${GA} \
                GEMBETA=${GB} NUMOFRUNS=${num_of_runs} NUMSTEPS=${num_steps} SUBSAMPLE=${sub_sample} \
                >>exports_${algorithm}.dat
              done
            done
          done
        done
      done
    done
  done
done

#!/bin/bash
##
## Copyright 2024 Max-Planck-Gesellschaft
## Code author: Michael Strecke, michael.strecke@tuebingen.mpg.de
## Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
## https://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##

BASE_COMMAND="python experiments/trajectory_fitting/optim_shapespace.py"

for toc_diff in {True,False}; do
  if [[ $toc_diff == "True" ]]; then
      toc_string=""
    else
      toc_string="no_toc_"
    fi
  for obj in {bob_and_spot,can,mug,camera}; do
    for ((seed=0; seed<$1; seed++)); do
      COMMENT="--comment=\"$(date -Idate)_trajectory_${toc_string}${obj}_${seed}\""

      ARGS="with name=${obj} use_toc_diff=${toc_diff} seed=${seed}"
      echo -e $BASE_COMMAND $COMMENT $ARGS
    done
  done
done

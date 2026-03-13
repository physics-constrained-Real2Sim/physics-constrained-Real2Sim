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

BASE_COMMAND="python experiments/system_identification/optim_sysid.py"

for target in {mass,force,friction}; do
  for ((seed=0; seed<$1; seed++)); do
    COMMENT="--comment=\"$(date -Idate)_sysid_${target}_${seed}\""

    ARGS="with ${target} seed=${seed}"
    echo -e $BASE_COMMAND $COMMENT $ARGS
  done
done

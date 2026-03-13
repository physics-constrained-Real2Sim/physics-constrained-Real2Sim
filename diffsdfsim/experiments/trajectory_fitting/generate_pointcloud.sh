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

BASE_COMMAND="python experiments/trajectory_fitting/optim_pointcloud.py"

for shape in {sphere,cube}; do
  for gravity in {True,False}; do
    for noise_fac in 0.0001; do
      if [[ $gravity == "True" ]]; then
        grav_comment="gr_on"
      else
        grav_comment="gr_off"
      fi
      for ((seed=0; seed<$1; seed++)); do
        COMMENT="--comment=\"$(date -Idate)_pointcloud_${shape}_${grav_comment}_${noise_fac}_${seed}\""

        ARGS="with shape=${shape} use_gravity=${gravity} depth_noise_factor=${noise_fac} strict_no_penetration=False seed=${seed}"
        echo -e $BASE_COMMAND $COMMENT $ARGS
      done
    done
  done
done

#
# Copyright 2024 Max-Planck-Gesellschaft
# Code author: Michael Strecke, michael.strecke@tuebingen.mpg.de
# Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import importlib
import json
import os
import sys

os.environ['PYOPENGL_PLATFORM'] = 'egl'

if __name__ == "__main__":
    if len(sys.argv) != 3 or not os.path.exists(sys.argv[1]):
        print("Usage: " + __file__ + "RUN_PATH OUTPUT_PATH, where RUN_PATH is the path to a completed run for the\n",
              "experiment and OUTPUT_PATH is the directory to write the output to.")
    run_dir = sys.argv[1]
    output_dir = sys.argv[2]
    run_file = os.path.join(run_dir, 'run.json')
    with open(run_file) as f:
        run = json.load(f)

    exp_mode = os.path.basename(run['experiment']['base_dir'])
    main_file = os.path.splitext(run['experiment']['mainfile'])[0]

    optim = importlib.import_module(".".join([exp_mode, main_file]))
    ex = optim.ex

    config_file = os.path.join(run_dir, 'config.json')
    ex.add_config(config_file)
    ex.run('record_results', config_updates={'run_dir': run_dir, 'output_dir': output_dir})

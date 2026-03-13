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
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 18})

basedir = Path(__file__).resolve().parent.parent.parent.parent
basedir = basedir.joinpath('experiments', 'trajectory_fitting_pointcloud')

pattern = '2024-04-24_pointcloud_{shape}_gr_{gr}_{noise_val}'

num_jobs_per_type = 20

shapes = ['sphere', 'cube']
grs = ['off', 'on']
noise_vals = [0.0001]


def read_data(basedir, pattern, shape, gr, noise_val):
    run_dirs = os.listdir(basedir)
    init_pos_errs = []
    frame_fit_pos_errs = []
    traj_fit_pos_errs = []
    init_rot_errs = []
    frame_fit_rot_errs = []
    traj_fit_rot_errs = []
    init_rads = []
    target_rads = []
    frame_fit_rads = []
    final_rads = []
    for run_dir in run_dirs:
        if re.match(r'[0-9]+', run_dir) is None:
            continue
        with open(os.path.join(basedir, run_dir, 'run.json')) as f:
            run = json.load(f)
        if re.match(pattern.format(shape=shape, gr=gr, noise_val=noise_val), run['meta']['comment']) is None:
            continue
        if not run['status'] == "COMPLETED":
            print("Run {} did not complete but exit with status: {}".format(run_dir, run['status']))
            continue
        with open(os.path.join(basedir, run_dir, 'metrics.json')) as f:
            metrics = json.load(f)

        with open(os.path.join(basedir, run_dir, 'output_first_frame.pkl'), 'rb') as f:
            data_first_frame = pickle.load(f)
        with open(os.path.join(basedir, run_dir, 'output_trajectory.pkl'), 'rb') as f:
            data_trajectory = pickle.load(f)

        init_pos_errs.append(metrics['frame_fit_pos_err']['values'][0])
        frame_fit_pos_errs.append(metrics['frame_fit_pos_err']['values'][-1])
        traj_fit_pos_errs.append(metrics['trajectory_fit_pos_err']['values'][-1])
        init_rot_errs.append(metrics['frame_fit_rot_err']['values'][0])
        frame_fit_rot_errs.append(metrics['frame_fit_rot_err']['values'][-1])
        traj_fit_rot_errs.append(metrics['trajectory_fit_rot_err']['values'][-1])
        init_rads.append(data_trajectory['start_rad'].item())
        target_rads.append(data_trajectory['target_rad'].item())
        frame_fit_rads.append(data_first_frame['final_rad'].item())
        final_rads.append(data_trajectory['final_rad'].item())

    assert len(init_pos_errs) == num_jobs_per_type

    init_pos_errs = np.array(init_pos_errs)
    frame_fit_pos_errs = np.array(frame_fit_pos_errs)
    traj_fit_pos_errs = np.array(traj_fit_pos_errs)
    init_rot_errs = np.array(init_rot_errs)
    frame_fit_rot_errs = np.array(frame_fit_rot_errs)
    traj_fit_rot_errs = np.array(traj_fit_rot_errs)
    init_rads = np.array(init_rads)
    target_rads = np.array(target_rads)
    frame_fit_rads = np.array(frame_fit_rads)
    final_rads = np.array(final_rads)

    return init_pos_errs, frame_fit_pos_errs, traj_fit_pos_errs, init_rot_errs, frame_fit_rot_errs, traj_fit_rot_errs, \
           init_rads, target_rads, frame_fit_rads, final_rads

table_dict = {}
for shape in shapes:
    table_dict[shape] = {}
    for gr in grs:
        table_dict[shape][gr] = {}
        for noise_val in noise_vals:
            init_pos_errs, frame_fit_pos_errs, traj_fit_pos_errs, init_rot_errs, frame_fit_rot_errs, traj_fit_rot_errs, \
            init_rads, target_rads, frame_fit_rads, final_rads = read_data(basedir, pattern, shape, gr, noise_val)

            if len(init_pos_errs) == 20:
                table_dict[shape][gr][str(noise_val)] = [
                   np.mean(init_pos_errs),
                   np.mean(frame_fit_pos_errs),
                   np.mean(traj_fit_pos_errs),
                   np.mean(init_rot_errs),
                   np.mean(frame_fit_rot_errs),
                   np.mean(traj_fit_rot_errs),
                   np.mean(np.abs(init_rads - target_rads)),
                   np.mean(np.abs(frame_fit_rads - target_rads)),
                   np.mean(np.abs(final_rads - target_rads))
                ]

                fig = plt.figure()
                plt.scatter(init_pos_errs, traj_fit_pos_errs)
                plt.plot([0, 0.1], [0, 0.1], ls='--', color='black')
                plt.xlabel('initial position error')
                plt.ylabel('result position error')
                if gr == 'on':
                    gr_text = 'w/'
                else:
                    gr_text = 'w/o'
                # plt.title('Init vs res pose error for {}, {} gravity'.format(shape, gr_text, noise_val))
                fig.tight_layout()
                fig.savefig('pointcloud_{}_{}.pdf'.format(shape, gr, noise_val))
            else:
                table_dict[shape][gr][str(noise_val)] = [0.] * 9


plt.show()

row_names = [
    'init pos',
    'pos frame fit',
    'pos traj. fit',
    'init rot',
    'rot frame fit',
    'rot traj. fit',
    'init size',
    'size frame fit',
    'size traj. fit'
]

for i, row_name in enumerate(row_names):
    print(row_name, end=' & ')
    for shape in shapes:
        for gr in grs:
            for noise_val in noise_vals:
                print('{:.3f}'.format(float(table_dict[shape][gr][str(noise_val)][i])), end=' ')
                if shape == shapes[-1] and gr == grs[-1] and noise_val == noise_vals[-1]:
                    print('\\\\')
                else:
                    print('&', end=' ')


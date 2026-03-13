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
import torch
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 14})

basedir = Path(__file__).resolve().parent.parent.parent.parent
sphere_basedir = basedir.joinpath('experiments', 'trajectory_fitting_sphere')

primitive_pattern = '2024-04-24_trajectory_sphere_gr_{gr}_toc_{toc}'

num_jobs_per_type = 50

gravity = ['on', 'off']
tocs = ['on', 'off']

fig, axs_primitives = plt.subplots(1, 2, figsize=(8, 4))


def read_data(basedir, pattern, gr, toc):
    run_dirs = os.listdir(basedir)
    chamfer_dists = []
    chamfer_steps = []
    objectives = []
    objective_steps = []
    start_radius_errs = []
    final_radius_errs = []
    for run_dir in run_dirs:
        if re.match(r'[0-9]+', run_dir) is None:
            continue
        with open(os.path.join(basedir, run_dir, 'run.json')) as f:
            run = json.load(f)
        if re.match(pattern.format(gr=gr, toc=toc), run['meta']['comment']) is None:
            continue
        if not run['status'] == "COMPLETED":
            print("Run {} for object {} did not complete but exit with status: {}".format(run_dir, name, run['status']))
        with open(os.path.join(basedir, run_dir, 'metrics.json')) as f:
            metrics = json.load(f)
        with open(os.path.join(basedir, run_dir, 'output.pkl'), 'rb') as f:
            data = pickle.load(f)

        start_radius_errs.append(data['target_rad'] - data['start_rad'])
        final_radius_errs.append(data['target_rad'] - data['final_rad'])
        chamfer_dists.append(metrics['chamfer_dist']['values'][:-1])
        chamfer_steps.append(metrics['chamfer_dist']['steps'][:-1])
        objectives.append(metrics['loss']['values'][:-1])
        objective_steps.append(metrics['loss']['steps'][:-1])

    assert len(chamfer_dists) == num_jobs_per_type
    max_len = max([len(cds) for cds in chamfer_dists])
    # max_len = 201
    chamfer_dists = [cds + [cds[-1]] * (max_len - len(cds)) for cds in chamfer_dists]
    chamfer_dists = torch.tensor(chamfer_dists)

    objectives = [objs + [objs[-1]] * (max_len - len(objs)) for objs in objectives]
    objectives = torch.tensor(objectives)

    start_radius_errs = torch.cat(start_radius_errs).detach().cpu()
    final_radius_errs = torch.cat(final_radius_errs).detach().cpu()

    return chamfer_dists, objectives, start_radius_errs, final_radius_errs


def plot_data(axs, dists, objs, names, sep=0.2, width=0.3):
    axs[0].boxplot([d[:, 0] for d in dists], positions=[i - sep for i in range(len(names))],
                   boxprops={'facecolor': 'C0'}, patch_artist=True, widths=[width] * len(names))
    axs[0].boxplot([d[:, -1] for d in dists], positions=[i + sep for i in range(len(names))],
                   boxprops={'facecolor': 'C2'}, patch_artist=True, widths=[width] * len(names))
    axs[0].set_xticks([i for i in range(len(names))])
    axs[0].set_xticklabels(names)
    axs[0].set_title('Chamfer distance')

    axs[1].boxplot([d[:, 0] for d in objs], positions=[i - sep for i in range(len(names))],
                   boxprops={'facecolor': 'C0'}, patch_artist=True, widths=[width] * len(names))
    axs[1].boxplot([d[:, -1] for d in objs], positions=[i + sep for i in range(len(names))],
                   boxprops={'facecolor': 'C2'}, patch_artist=True, widths=[width] * len(names))
    axs[1].set_xticks([i for i in range(len(names))])
    axs[1].set_xticklabels(names)
    axs[1].set_title('Objective values')


dists, objs = [], []
names = []

for gr in gravity:
    for toc in tocs:
        chamfer_dist, objectives, start_rad_errs, final_rad_errs = read_data(sphere_basedir, primitive_pattern, gr, toc)
        dists.append(chamfer_dist)
        objs.append(objectives)
        if gr == 'on':
            name = 'double bounce'
        else:
            name = 'single bounce'
        if toc == 'on':
            name += '\ntoc'
        else:
            name += '\nno toc'
        names.append(name)

        print('gr: {}, toc: {}: radius error (min, mean, max): {} {} {}'.format(gr, toc, final_rad_errs.abs().min(),
                                                                                final_rad_errs.abs().mean(),
                                                                                final_rad_errs.abs().max()))

        if gr == 'on' and toc == 'on':
            start_errs_w_toc = start_rad_errs
            final_errs_w_toc = final_rad_errs
            objs_w_toc = objectives
        elif gr == 'on' and toc == 'off':
            start_errs_wo_toc = start_rad_errs
            final_errs_wo_toc = final_rad_errs
            objs_wo_toc = objectives

plot_data(axs_primitives, dists, objs, names)

fig2, ax = plt.subplots(1, 2, figsize=(8, 3))
# ax[0].scatter(start_rads, final_rads)
# ax[0].scatter(start_rads_no_toc, final_rads_no_toc)
ax[0].scatter(start_errs_w_toc, final_errs_w_toc)
ax[0].scatter(start_errs_wo_toc, final_errs_wo_toc)
ax[0].set_xlabel('start radius error')
ax[0].set_ylabel('result radius error')
ax[0].legend(['w/ toc', 'w/o toc'])

losses = objs_w_toc
q = losses.quantile(torch.tensor([0, 0.25, 0.5, 0.75, 1]), dim=0)

l = ax[1].plot(q[2])
ax[1].fill_between(range(q[1].shape[0]), q[1], q[3], alpha=0.15, color=l[0].get_color())
ax[1].plot(q[0], color=l[0].get_color(), linestyle='--', linewidth=1.0)
ax[1].plot(q[4], color=l[0].get_color(), linestyle='--', linewidth=1.0)

losses_no_toc = objs_wo_toc
q = losses_no_toc.quantile(torch.tensor([0, 0.25, 0.5, 0.75, 1]), dim=0)

l = ax[1].plot(q[2])
ax[1].fill_between(range(q[1].shape[0]), q[1], q[3], alpha=0.15, color=l[0].get_color())
ax[1].plot(q[0], color=l[0].get_color(), linestyle='--', linewidth=1.0)
ax[1].plot(q[4], color=l[0].get_color(), linestyle='--', linewidth=1.0)

ax[1].set_xlabel('iteration')
ax[1].set_ylabel('objective')

fig.tight_layout()
fig2.tight_layout()
plt.show()
fig.savefig('sphere_trajectory.pdf')
fig2.savefig('sphere_toc_analysis.pdf')

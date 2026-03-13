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

# 'font.serif': 'Times New Roman', 'font.family': 'serif',
plt.rcParams.update({'font.size': 12, 'figure.figsize': (8, 4)})

basedir = Path(__file__).resolve().parent.parent.parent.parent
basedir = basedir.joinpath('experiments', 'trajectory_fitting_shapespace')

pattern = '2024-04-24_trajectory_{name}'
no_toc_pattern = '2024-04-24_trajectory_no_toc_{name}'

shapespaces = ['bob_and_spot', 'can', 'camera', 'mug']

num_jobs_per_name = 50


def read_data(basedir, pattern, name):
    run_dirs = os.listdir(basedir)
    chamfer_dists = []
    chamfer_steps = []
    objectives = []
    objective_steps = []
    last_step_dists = []
    for run_dir in run_dirs:
        if re.match(r'[0-9]+', run_dir) is None:
            continue
        with open(os.path.join(basedir, run_dir, 'run.json')) as f:
            run = json.load(f)
        if re.match(pattern.format(name=name), run['meta']['comment']) is None:
            continue
        if not run['status'] == "COMPLETED":
            print("Run {} for object {} did not complete but exit with status: {}".format(run_dir, name, run['status']))
            continue
        with open(os.path.join(basedir, run_dir, 'metrics.json')) as f:
            metrics = json.load(f)

        chamfer_dists.append(metrics['chamfer_dist']['values'][:-1])
        chamfer_steps.append(metrics['chamfer_dist']['steps'][:-1])
        objectives.append(metrics['loss']['values'][:-1])
        objective_steps.append(metrics['loss']['steps'][:-1])

        with open(os.path.join(basedir, run_dir, 'output.pkl'), 'rb') as f:
            data = pickle.load(f)

        last_step_dists.append([(data['init_trajectory'][-1][1][-3:] - data['target_trajectory'][-1][1][-3:]).norm().item(),
                                (data['trajectory'][-1][1][-3:] - data['target_trajectory'][-1][1][-3:]).norm().item()])

    # One of the no_toc runs ended up in an invalid state and failed.
    assert ((len(chamfer_dists) == num_jobs_per_name)
            or (name == 'mug' and 'no_toc' in pattern and len(chamfer_dists) == num_jobs_per_name - 1))

    max_len = max([len(cds) for cds in chamfer_dists])
    # max_len = 201
    chamfer_dists = [cds + [cds[-1]] * (max_len - len(cds)) for cds in chamfer_dists]
    chamfer_dists = np.array(chamfer_dists)

    objectives = [objs + [objs[-1]] * (max_len - len(objs)) for objs in objectives]
    objectives = np.array(objectives)

    last_step_dists = np.array(last_step_dists)

    return chamfer_dists, objectives, last_step_dists


def plot_data(axs, dists, objs, names, sep=0.2, width=0.3, show_outliers=False, scale_result=False):
    axs[0].boxplot([d[:, 0] for d in dists], positions=[i - sep for i in range(len(names))],
                   boxprops={'facecolor': 'C0'}, patch_artist=True, widths=[width] * len(names),
                   showfliers=show_outliers)
    res = axs[0].boxplot([d[:, -1] for d in dists], positions=[i + sep for i in range(len(names))],
                   boxprops={'facecolor': 'C2'}, patch_artist=True, widths=[width] * len(names))
    if scale_result:
        ymin = np.min(np.concatenate([w.get_data()[1] for w in res['whiskers']]))
        ymax = np.max(np.concatenate([w.get_data()[1] for w in res['whiskers']]))
        axs[0].set_ylim((ymin - 0.05*(ymax - ymin), ymax + 0.05*(ymax - ymin)))
    axs[0].set_xticks([i for i in range(len(names))])
    axs[0].set_xticklabels([n.replace('_', ' ') for n in names])
    axs[0].set_title('Chamfer distance')

    axs[1].boxplot([d[:, 0] for d in objs], positions=[i - sep for i in range(len(names))],
                   boxprops={'facecolor': 'C0'}, patch_artist=True, widths=[width] * len(names),
                   showfliers=show_outliers)
    res = axs[1].boxplot([d[:, -1] for d in objs], positions=[i + sep for i in range(len(names))],
                   boxprops={'facecolor': 'C2'}, patch_artist=True, widths=[width] * len(names))
    if scale_result:
        ymin = np.min(np.concatenate([w.get_data()[1] for w in res['whiskers']]))
        ymax = np.max(np.concatenate([w.get_data()[1] for w in res['whiskers']]))
        axs[1].set_ylim((ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin)))
    axs[1].set_xticks([i for i in range(len(names))])
    axs[1].set_xticklabels([n.replace('_', ' ') for n in names])
    axs[1].set_title('Objective values')


def print_table(name, dists_shapespaces, no_toc_dists):
    print('\\multirow{6}{*}{\\rotatebox{90}{' + name + '}}', end=' & ')
    print('mean', end=' & ')
    for dist, no_toc_dist in zip(dists_shapespaces, no_toc_dists):
        print('{:.4f} & {:.4f} & {:.4f}'.format(np.mean(dist[:, 0]), np.mean(dist[:, -1]), np.mean(no_toc_dist[:, -1])),
              end=' ')
        if dist is dists_shapespaces[-1]:
            print('\\\\\\cmidrule(lr){2-14}')
        else:
            print('&', end=' ')
    print('& min', end=' & ')
    for dist, no_toc_dist in zip(dists_shapespaces, no_toc_dists):
        print('{:.4f} & {:.4f} & {:.4f}'.format(np.min(dist[:, 0]), np.min(dist[:, -1]), np.min(no_toc_dist[:, -1])),
              end=' ')
        if dist is dists_shapespaces[-1]:
            print('\\\\')
        else:
            print('&', end=' ')
    print('& Q25', end=' & ')
    for dist, no_toc_dist in zip(dists_shapespaces, no_toc_dists):
        print('{:.4f} & {:.4f} & {:.4f}'.format(np.percentile(dist[:, 0], 25), np.percentile(dist[:, -1], 25),
                                                np.percentile(no_toc_dist[:, -1], 25)), end=' ')
        if dist is dists_shapespaces[-1]:
            print('\\\\')
        else:
            print('&', end=' ')
    print('& median', end=' & ')
    for dist, no_toc_dist in zip(dists_shapespaces, no_toc_dists):
        print('{:.4f} & {:.4f} & {:.4f}'.format(np.median(dist[:, 0]), np.median(dist[:, -1]),
                                                np.median(no_toc_dist[:, -1])), end=' ')
        if dist is dists_shapespaces[-1]:
            print('\\\\')
        else:
            print('&', end=' ')
    print('& Q75', end=' & ')
    for dist, no_toc_dist in zip(dists_shapespaces, no_toc_dists):
        print('{:.4f} & {:.4f} & {:.4f}'.format(np.percentile(dist[:, 0], 75), np.percentile(dist[:, -1], 75),
                                                np.percentile(no_toc_dist[:, -1], 75)), end=' ')
        if dist is dists_shapespaces[-1]:
            print('\\\\')
        else:
            print('&', end=' ')
    print('& max', end=' & ')
    for dist, no_toc_dist in zip(dists_shapespaces, no_toc_dists):
        print('{:.4f} & {:.4f} & {:.4f}'.format(np.max(dist[:, 0]), np.max(dist[:, -1]), np.max(no_toc_dist[:, -1])),
              end=' ')
        if dist is dists_shapespaces[-1]:
            print('\\\\\\midrule')
        else:
            print('&', end=' ')


dists, objs, last_step_dists = [], [], []
for shapespace in shapespaces:
    chamfer_dist, objectives, last_step_d = read_data(basedir, pattern, shapespace)
    dists.append(chamfer_dist)
    objs.append(objectives)
    last_step_dists.append(last_step_d)

no_toc_dists, no_toc_objs, no_toc_last_step_dists = [], [], []
for shapespace in shapespaces:
    chamfer_dist, objectives, last_step_d = read_data(basedir, no_toc_pattern, shapespace)
    no_toc_dists.append(chamfer_dist)
    no_toc_objs.append(objectives)
    no_toc_last_step_dists.append(last_step_d)

fig, axs_shapespace = plt.subplots(1, 2)
plot_data(axs_shapespace, dists, objs, shapespaces, width=0.3)
fig.tight_layout()
fig.savefig('trajectory_shapespace.pdf')

fig, axs_shapespace = plt.subplots(1, 2)
plot_data(axs_shapespace, dists, objs, shapespaces, width=0.3, show_outliers=True)
fig.tight_layout()
fig.savefig('trajectory_shapespace_outliers.pdf')

fig, axs_shapespace = plt.subplots(1, 2)
plot_data(axs_shapespace, dists, objs, shapespaces, width=0.3, show_outliers=True, scale_result=True)
fig.tight_layout()
fig.savefig('trajectory_shapespace_result_scale.pdf')

fig, axs_shapespace = plt.subplots(1, 2)
plot_data(axs_shapespace, no_toc_dists, no_toc_objs, shapespaces, width=0.3)
fig.tight_layout()
fig.savefig('trajectory_shapespace_no_toc.pdf')

fig, axs_shapespace = plt.subplots(1, 2)
plot_data(axs_shapespace, no_toc_dists, no_toc_objs, shapespaces, width=0.3, show_outliers=True)
fig.tight_layout()
fig.savefig('trajectory_shapespace_no_toc_outliers.pdf')

fig, axs_shapespace = plt.subplots(1, 2)
plot_data(axs_shapespace, no_toc_dists, no_toc_objs, shapespaces, width=0.3, show_outliers=True, scale_result=True)
fig.tight_layout()
fig.savefig('trajectory_shapespace_no_toc_result_scale.pdf')

print_table('CD', dists, no_toc_dists)
print_table('obj', objs, no_toc_objs)
print_table('pos err', last_step_dists, no_toc_last_step_dists)

mean_dist = np.asanyarray([(d[:, 0], d[:, -1]) for d in dists]).mean(axis=0).mean(axis=1)
mean_dist_no_toc = sum([d[:, -1].mean() for d in no_toc_dists]) / len(no_toc_dists)
print("Mean CD init: {}, result: {}, no toc: {}".format(mean_dist[0], mean_dist[1], mean_dist_no_toc))

mean_obj = np.asanyarray([(d[:, 0], d[:, -1]) for d in objs]).mean(axis=0).mean(axis=1)
mean_obj_no_toc = sum([d[:, -1].mean() for d in no_toc_objs]) / len(no_toc_objs)
print("Mean obj init: {}, result: {}, no toc: {}".format(mean_obj[0], mean_obj[1], mean_obj_no_toc))

mean_pos = np.asanyarray([(d[:, 0], d[:, -1]) for d in last_step_dists]).mean(axis=0).mean(axis=1)
mean_pos_no_toc = sum([d[:, -1].mean() for d in no_toc_last_step_dists]) / len(no_toc_last_step_dists)
print("Mean CD init: {}, result: {}, no toc: {}".format(mean_pos[0], mean_pos[1], mean_pos_no_toc))

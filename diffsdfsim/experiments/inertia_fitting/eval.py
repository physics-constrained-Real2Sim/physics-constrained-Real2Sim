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
import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

# 'font.serif': 'Times New Roman', 'font.family': 'serif',
plt.rcParams.update({'font.size': 12, 'figure.figsize': (8, 4)})

basedir = Path(__file__).resolve().parent.parent.parent.parent
primitive_basedir = basedir.joinpath('experiments', 'inertia_fitting_primitives')
shapespace_basedir = basedir.joinpath('experiments', 'inertia_fitting_shapespace')

primitive_pattern = '2024-04-24_inertia_{name}'
shapespace_pattern = '2024-04-24_inertia_{name}'

primitives = ['box', 'sphere', 'cylinder']
shapespaces = ['bob_and_spot', 'can', 'camera', 'mug']

num_jobs_per_name = 50

def read_data(basedir, pattern, name):
    run_dirs = os.listdir(basedir)
    chamfer_dists = []
    chamfer_steps = []
    objectives = []
    objective_steps = []
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

        chamfer_dists.append(metrics['chamfer_dist']['values'])
        chamfer_steps.append(metrics['chamfer_dist']['steps'])
        objectives.append(metrics['loss']['values'])
        objective_steps.append(metrics['loss']['steps'])

    assert len(chamfer_dists) == num_jobs_per_name
    max_len = max([len(cds) for cds in chamfer_dists])
    # max_len = 201
    chamfer_dists = [cds + [cds[-1]] * (max_len - len(cds)) for cds in chamfer_dists]
    chamfer_dists = np.array(chamfer_dists)

    objectives = [objs + [objs[-1]] * (max_len - len(objs)) for objs in objectives]
    objectives = np.array(objectives)

    return chamfer_dists, objectives


def plot_data(axs, dists, objs, names, sep=0.2, width=0.3, show_outliers=False, scale_result=False):
    axs[0].boxplot([d[:, 0] for d in dists], positions=[i - sep for i in range(len(names))],
                   boxprops={'facecolor': 'C0'}, patch_artist=True, widths=[width] * len(names),
                   showfliers=show_outliers)
    res = axs[0].boxplot([d[:, -1] for d in dists], positions=[i + sep for i in range(len(names))],
                         boxprops={'facecolor': 'C2'}, patch_artist=True, widths=[width] * len(names))
    if scale_result:
        ymin = np.min(np.concatenate([w.get_data()[1] for w in res['whiskers']]))
        ymax = np.max(np.concatenate([w.get_data()[1] for w in res['whiskers']]))
        axs[0].set_ylim((ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin)))
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


def print_table(name, dists_primitives, dists_shapespaces):
    print('\\multirow{6}{*}{\\rotatebox{90}{' + name + '}}', end=' & ')
    print('mean', end=' & ')
    for dist in dists_primitives:
        print('{:.1e} & {:.1e}'.format(np.mean(dist[:, 0]), np.mean(dist[:, -1])), end=' & ')
    for dist in dists_shapespaces:
        print('{:.1e} & {:.1e}'.format(np.mean(dist[:, 0]), np.mean(dist[:, -1])), end=' ')
        if dist is dists_shapespaces[-1]:
            print('\\\\\\cmidrule(lr){2-16}')
        else:
            print('&', end=' ')
    print('& min', end=' & ')
    for dist in dists_primitives:
        print('{:.1e} & {:.1e}'.format(np.min(dist[:, 0]), np.min(dist[:, -1])), end=' & ')
    for dist in dists_shapespaces:
        print('{:.1e} & {:.1e}'.format(np.min(dist[:, 0]), np.min(dist[:, -1])), end=' ')
        if dist is dists_shapespaces[-1]:
            print('\\\\')
        else:
            print('&', end=' ')
    print('& Q25', end=' & ')
    for dist in dists_primitives:
        print('{:.1e} & {:.1e}'.format(np.percentile(dist[:, 0], 25), np.percentile(dist[:, -1], 25)), end=' & ')
    for dist in dists_shapespaces:
        print('{:.1e} & {:.1e}'.format(np.percentile(dist[:, 0], 25), np.percentile(dist[:, -1], 25)), end=' ')
        if dist is dists_shapespaces[-1]:
            print('\\\\')
        else:
            print('&', end=' ')
    print(' & median', end=' & ')
    for dist in dists_primitives:
        print('{:.1e} & {:.1e}'.format(np.median(dist[:, 0]), np.median(dist[:, -1])), end=' & ')
    for dist in dists_shapespaces:
        print('{:.1e} & {:.1e}'.format(np.median(dist[:, 0]), np.median(dist[:, -1])), end=' ')
        if dist is dists_shapespaces[-1]:
            print('\\\\')
        else:
            print('&', end=' ')
    print('& Q75', end=' & ')
    for dist in dists_primitives:
        print('{:.1e} & {:.1e}'.format(np.percentile(dist[:, 0], 75), np.percentile(dist[:, -1], 75)), end=' & ')
    for dist in dists_shapespaces:
        print('{:.1e} & {:.1e}'.format(np.percentile(dist[:, 0], 75), np.percentile(dist[:, -1], 75)), end=' ')
        if dist is dists_shapespaces[-1]:
            print('\\\\')
        else:
            print('&', end=' ')
    print('& max', end=' & ')
    for dist in dists_primitives:
        print('{:0.0e} & {:.1e}'.format(np.max(dist[:, 0]), np.max(dist[:, -1])), end=' & ')
    for dist in dists_shapespaces:
        print('{:.1e} & {:.1e}'.format(np.max(dist[:, 0]), np.max(dist[:, -1])), end=' ')
        if dist is dists_shapespaces[-1]:
            print('\\\\\\midrule')
        else:
            print('&', end=' ')


dists_primitives, objs_primitives = [], []
for primitive in primitives:
    chamfer_dist, objectives = read_data(primitive_basedir, primitive_pattern, primitive)
    dists_primitives.append(chamfer_dist)
    objs_primitives.append(objectives)

dists_shapespaces, objs_shapespaces = [], []
for shapespace in shapespaces:
    chamfer_dist, objectives = read_data(shapespace_basedir, shapespace_pattern, shapespace)
    dists_shapespaces.append(chamfer_dist)
    objs_shapespaces.append(objectives)

fig_primitives, axs_primitives = plt.subplots(1, 2)
plot_data(axs_primitives, dists_primitives, objs_primitives, primitives)
fig_primitives.tight_layout()
fig_primitives.savefig('inertia_primitives.pdf')

fig_shapespace, axs_shapespace = plt.subplots(1, 2)
plot_data(axs_shapespace, dists_shapespaces, objs_shapespaces, shapespaces, width=0.3)
fig_shapespace.tight_layout()
fig_shapespace.savefig('inertia_shapespace.pdf')

fig_primitives, axs_primitives = plt.subplots(1, 2)
plot_data(axs_primitives, dists_primitives, objs_primitives, primitives, show_outliers=True)
fig_primitives.tight_layout()
fig_primitives.savefig('inertia_primitives_outliers.pdf')

fig_shapespace, axs_shapespace = plt.subplots(1, 2)
plot_data(axs_shapespace, dists_shapespaces, objs_shapespaces, shapespaces, width=0.3, show_outliers=True)
fig_shapespace.tight_layout()
fig_shapespace.savefig('inertia_shapespace_outliers.pdf')

fig_primitives, axs_primitives = plt.subplots(1, 2)
plot_data(axs_primitives, dists_primitives, objs_primitives, primitives, show_outliers=True, scale_result=True)
fig_primitives.tight_layout()
fig_primitives.savefig('inertia_primitives_result_scale.pdf')

fig_shapespace, axs_shapespace = plt.subplots(1, 2)
plot_data(axs_shapespace, dists_shapespaces, objs_shapespaces, shapespaces, width=0.3, show_outliers=True,
          scale_result=True)
fig_shapespace.tight_layout()
fig_shapespace.savefig('inertia_shapespace_result_scale.pdf')

print_table('CD', dists_primitives, dists_shapespaces)
print_table('obj', objs_primitives, objs_shapespaces)

mean_dist = np.asanyarray([(d[:, 0], d[:, -1]) for d in dists_primitives + dists_shapespaces]).mean(axis=0).mean(axis=1)
print("Mean CD init: {}, result: {}".format(mean_dist[0], mean_dist[1]))

mean_obj = np.asanyarray([(d[:, 0], d[:, -1]) for d in objs_primitives + objs_shapespaces]).mean(axis=0).mean(axis=1)
print("Mean obj init: {}, result: {}".format(mean_obj[0], mean_obj[1]))

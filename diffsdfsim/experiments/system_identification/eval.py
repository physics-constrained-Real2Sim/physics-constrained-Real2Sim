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

plt.rcParams.update({'font.size': 12, 'figure.figsize': (8, 4)})

basedir = Path(__file__).resolve().parent.parent.parent.parent
basedir = basedir.joinpath('experiments', 'system_identification')

pattern = '2024-04-24_sysid_{type}'

types = ['mass', 'force', 'friction']

num_jobs_per_type = 50

def read_data(basedir, pattern, type):
    run_dirs = os.listdir(basedir)
    dists = []
    dist_steps = []
    objectives = []
    objective_steps = []
    for run_dir in run_dirs:
        if re.match(r'[0-9]+', run_dir) is None:
            continue
        with open(os.path.join(basedir, run_dir, 'run.json')) as f:
            run = json.load(f)
        if re.match(pattern.format(type=type), run['meta']['comment']) is None:
            continue
        if not run['status'] == "COMPLETED":
            print("Run {} for object {} did not complete but exit with status: {}".format(run_dir, type, run['status']))
            continue
        with open(os.path.join(basedir, run_dir, 'metrics.json')) as f:
            metrics = json.load(f)

        dists.append(metrics['dist']['values'])
        dist_steps.append(metrics['dist']['steps'])
        objectives.append(metrics['loss']['values'])
        objective_steps.append(metrics['loss']['steps'])

    assert len(dists) == num_jobs_per_type
    max_len = max([len(cds) for cds in dists])
    # max_len = 201
    dists = [cds + [cds[-1]] * (max_len - len(cds)) for cds in dists]
    dists = np.array(dists)

    objectives = [objs + [objs[-1]] * (max_len - len(objs)) for objs in objectives]
    objectives = np.array(objectives)

    return dists, objectives


def plot_data(axs, dists, objs, types, sep=0.2, width=0.3, scale_result=False):
    axs[0].boxplot([d[:, 0] for d in dists], positions=[i - sep for i in range(len(types))],
                   boxprops={'facecolor': 'C0'}, patch_artist=True, widths=[width] * len(types))
    res = axs[0].boxplot([d[:, -1] for d in dists], positions=[i + sep for i in range(len(types))],
                         boxprops={'facecolor': 'C2'}, patch_artist=True, widths=[width] * len(types))
    if scale_result:
        ymin = np.min(np.concatenate([w.get_data()[1] for w in res['whiskers']]))
        ymax = np.max(np.concatenate([w.get_data()[1] for w in res['whiskers']]))
        axs[0].set_ylim((ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin)))
    axs[0].set_xticks([i for i in range(len(types))])
    axs[0].set_xticklabels([n.replace('_', ' ') for n in types])
    axs[0].set_title('Target error')

    axs[1].boxplot([d[:, 0] for d in objs], positions=[i - sep for i in range(len(types))],
                   boxprops={'facecolor': 'C0'}, patch_artist=True, widths=[width] * len(types))
    res = axs[1].boxplot([d[:, -1] for d in objs], positions=[i + sep for i in range(len(types))],
                         boxprops={'facecolor': 'C2'}, patch_artist=True, widths=[width] * len(types))
    if scale_result:
        ymin = np.min(np.concatenate([w.get_data()[1] for w in res['whiskers']]))
        ymax = np.max(np.concatenate([w.get_data()[1] for w in res['whiskers']]))
        axs[1].set_ylim((ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin)))
    axs[1].set_xticks([i for i in range(len(types))])
    axs[1].set_xticklabels([n.replace('_', ' ') for n in types])
    axs[1].set_title('Objective values')


def print_table(name, dists):
    print('\\multirow{6}{*}{\\rotatebox{90}{\parbox{3em}{' + name + '}}}', end=' & ')
    print('mean', end=' & ')
    for dist in dists:
        print('{:.1e} & {:.1e}'.format(np.mean(dist[:, 0]), np.mean(dist[:, -1])), end=' ')
        if dist is dists[-1]:
            print('\\\\\\cmidrule(lr){2-8}')
        else:
            print('&', end=' ')
    print('& min', end=' & ')
    for dist in dists:
        print('{:.1e} & {:.1e}'.format(np.min(dist[:, 0]), np.min(dist[:, -1])), end=' ')
        if dist is dists[-1]:
            print('\\\\')
        else:
            print('&', end=' ')
    print('& Q25', end=' & ')
    for dist in dists:
        print('{:.1e} & {:.1e}'.format(np.percentile(dist[:, 0], 25), np.percentile(dist[:, -1], 25)), end=' ')
        if dist is dists[-1]:
            print('\\\\')
        else:
            print('&', end=' ')
    print('& median', end=' & ')
    for dist in dists:
        print('{:.1e} & {:.1e}'.format(np.median(dist[:, 0]), np.median(dist[:, -1])), end=' ')
        if dist is dists[-1]:
            print('\\\\')
        else:
            print('&', end=' ')
    print('& Q75', end=' & ')
    for dist in dists:
        print('{:.1e} & {:.1e}'.format(np.percentile(dist[:, 0], 75), np.percentile(dist[:, -1], 75)), end=' ')
        if dist is dists[-1]:
            print('\\\\')
        else:
            print('&', end=' ')
    print('& max', end=' & ')
    for dist in dists:
        print('{:.1e} & {:.1e}'.format(np.max(dist[:, 0]), np.max(dist[:, -1])), end=' ')
        if dist is dists[-1]:
            print('\\\\\\midrule')
        else:
            print('&', end=' ')


dists, objs = [], []
for type in types:
    dist, objectives = read_data(basedir, pattern, type)
    dists.append(dist)
    objs.append(objectives)

    print('{}: average error init: {} average error result: {}'.format(type, np.mean(dist[:, 0]), np.mean(dist[:, -1])))

fig, axs = plt.subplots(1, 2)
plot_data(axs, dists, objs, types)
fig.tight_layout()
fig.savefig('system_id.pdf')

fig, axs = plt.subplots(1, 2)
plot_data(axs, dists, objs, types, scale_result=True)
fig.tight_layout()
fig.savefig('system_id_result_scale.pdf')

print_table('target error', dists)
print_table('objective', objs)

# plt.show()

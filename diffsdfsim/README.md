# DiffSDFSim

This repository provides source code for DiffSDFSim accompanying the following publication:

*Michael Strecke and Joerg Stueckler, "**DiffSDFSim: Differentiable Rigid-Body Dynamics With Implicit Shapes**"*  
*Presented at the **International Conference on 3D Vision (3DV) 2021***

Please see the [project homepage](https://diffsdfsim.is.tue.mpg.de/) for details.

If you use the source code provided in this repository for your research, please cite the corresponding publication as:
```
@inproceedings{strecke2021_diffsdfsim,
  title = {{DiffSDFSim}: Differentiable Rigid-Body Dynamics With Implicit Shapes},
  author = {Strecke, Michael and Stueckler, Joerg},
  booktitle = {International Conference on {3D} Vision ({3DV})},
  month = dec,
  year = {2021},
  doi = {10.1109/3DV53792.2021.00020},
  month_numeric = {12}
}
```

## Getting started

### Folder structure for experiments
Set up the following folder structure for the experiments:
```
diffsdfsim/ <---- project root
|-experiments/ <---- folder for experiment results
|  |-inertia_fitting_primitives/
|  |-inertia_fitting_shapespace/
|  |- ....
|-shapespaces/ <---- trained IGR shapespaces
|-IGR/        <---- the IGR repo
`-diffsdfsim/ <---- this repo
```

I.e. create a project root folder and clone this repository, as well as the [IGR repo](https://github.com/amosgropp/IGR)
into it.
Further, create directories for the experiment results and the learned shapespaces directly inside the project root.
We provide model weights for IGR for the "bob and spot" shape space in [this KEEPER repository](https://keeper.mpdl.mpg.de/d/c446493994b5486a843a/),
which you can copy to the project root.
As the other learned shape spaces were trained on ShapeNet, we are unable to directly provide the model weights without
users agreeing to ShapeNet's [terms of use](https://shapenet.org/terms).
Instructions on how we preprocessed and trained IGR can be obtained in [TRAIN_IGR.md](TRAIN_IGR.md).

If you clone IGR to a different location, make sure to set the environment variable `IGR_PATH` to the correct path before
running the experiments.

### Set up conda environment using a recent PyTorch version
1. Install CUDA 12.1
2. Setup and activate conda environment:
   ```bash
   export PIP_NO_BUILD_ISOLATION=0
   # export TORCH_CUDA_ARCH_LIST="..."  # Optionally set the list of CUDA architectures you want the code to run on, see https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.CUDAExtension 
   conda env create -f environment.yaml
   conda activate diffdsdfsim
   ```
   The `PIP_NO_BUILD_ISOLATION` environment variable makes sure the build of the [ev_sdf_utils](https://github.com/EmbodiedVision/ev-sdf-utils) package uses the same
   PyTorch version as you have in your environment.
3. Add the path of this repository to your `PYTHONPATH`
   ```bash
   export PYTHONPATH=$PWD
   ```

### Legacy environment setup with CUDA 10.2 and older Pytorch3D version to exactly reproduce paper results

The results in the paper were obtained with Pytorch3D version 0.4.0.
Numerical changes introduced in version 0.5.0 yield different optimization results (especially noticeable in the
"bob and spot on pole" demo).
Unfortunately, we experienced issues running our code in an environment with Pytorch 1.10 (the latest version compatible
with Pytorch3D 0.4.0) on recent GPU hardware (RTX 4090).
To run the code with Pytorch3D 0.4.0, you need a GPU that supports CUDA 10.2 (i.e. compute capability < 8.0).
If you have such a GPU, you can follow these instructions for results closer to those reported in the paper.

1. Install CUDA 10.2
2. Download the CUB library:
   ```bash
   curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
   tar xzf 1.10.0.tar.gz
   export CUB_HOME=$PWD/cub-1.10.0
   ```
3. Setup the conda environment:
   ```bash
   export PIP_NO_BUILD_ISOLATION=0
   # export TORCH_CUDA_ARCH_LIST="..."  # Optionally set the list of CUDA architectures you want the code to run on, see https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.CUDAExtension
   conda env create -f environment_cu102.yaml
   conda activate diffsdfsim_cu102
   ```
4. Downgrade setuptools to avoid `AttributeError: module 'distutils' has no attribute 'version'`
   ```bash
   conda install setuptools=59.5.0 -c conda-forge
   ``` 

## Running the experiments
### Demo
Run the following script to run the demo dropping the cow (spot) on the pole and optimizing it for the genus-1 duck (bob):
```bash
python demos/demo_meshsdf.py
```
The script will write visualizations for all iterations and a binary file containing the latent codes and losses for all 
optimization steps to the experiments directory mentioned above.

The default version will work with recent versions of Pytorch (2.1.0) and Pytorch3D (0.7.5).
To exactly reproduce the optimization that generated the paper results, set up the environment with the workaround
mentioned above and change the `LEGACY` variable in [demos/demo_meshsdf.py](demos/demo_meshsdf.py) to `True`.

### Quantitative Experiments
To reproduce the quantitative experiments, run the scripts inside the [experiments](experiments) directory.
The scripts use [sacred](https://github.com/IDSIA/sacred) for configuration and storing results.
If you installed the conda environment with more recent PyTorch and Pytorch3D versions, there might be numerical
differences in the numbers compared to the ones reported in the paper.
To exactly reproduce the paper results, you can follow the
[legacy environment setup](#legacy-environment-setup-with-cuda-102-and-older-pytorch3d-version-to-exactly-reproduce-paper-results)
above to set up an environment with the software versions that produced the paper results.
The changes in the evaluation numbers are very small.
With the [more recent setup](#set-up-conda-environment-using-a-recent-pytorch-version) described above, one should be able to reproduce the numbers as in [RESULTS.md](RESULTS.md).

#### Bouncing sphere
A list of jobs for the bouncing sphere experiment can be generated by
```bash
bash experiments/trajectory_fitting/generete_sphere.sh 50
```
After running the jobs, the evaluation script [experiments/trajectory_fitting/eval_sphere.py](experiments/trajectory_fitting/eval_sphere.py)
can be run to generate figures and tables from the paper.
You will need to adapt the date in the `primitive_pattern` string of the script first.

#### Bouncing Shapespace
A list of jobs for the bouncing shapespace experiment can be generated by
```bash
bash experiments/trajectory_fitting/generete_shapespace.sh 50
```
After running the jobs, the evaluation script [experiments/trajectory_fitting/eval_shapespace.py](experiments/trajectory_fitting/eval_shapespace.py)
can be run to generate figures and tables from the paper.
You will need to adapt the dates in the `pattern` and `no_toc_pattern` strings of the script first.

To run all the jobs, you will need IGR model weights for the can, camera, and mug categories from ShapeNetV1. See
[TRAIN_IGR.md](TRAIN_IGR.md) for a description on how we preprocessed and trained IGR.

#### Fitting to depth observations
A list of jobs for the experiment fitting to depth observations can be generated by
```bash
bash experiments/trajectory_fitting/generete_pointcloud.sh 20
```
After running the jobs, the evaluation script [experiments/trajectory_fitting/eval_pointcloud.py](experiments/trajectory_fitting/eval_pointcloud.py)
can be run to generate figures and tables from the paper.
You will need to adapt the date in the `pattern` string of the script first.


#### Shape from Inertia
Lists of jobs for the shape from inertia experiments can be generated by
```bash
bash experiments/inertia_fitting/generete_primitives.sh 50
bash experiments/inertia_fitting/generete_shapespace.sh 50
```
After running the jobs, the evaluation script [experiments/inertia_fitting/eval.py](experiments/inertia_fitting/eval.py)
can be run to generate figures and tables from the paper.
You will need to adapt the date in the `primitive_pattern` and `shapespace_pattern` strings of the script first.

To run all the jobs, you will need IGR model weights for the can, camera, and mug categories from ShapeNetV1. See [TRAIN_IGR.md](TRAIN_IGR.md)
for a description on how we preprocessed and trained IGR.

#### System Identification
A list of jobs for the system identification experiment can be generated by
```bash
bash experiments/system_identification/generate_sysid.sh 50
```
After running the jobs, the evaluation script [experiments/system_identification/eval.py](experiments/system_identification/eval.py)
can be run to generate figures and tables from the paper.
You will need to adapt the date in the `pattern` string of the script first.

The system identification results in the paper were obtained with different model weights for the "bob and spot"
shapespace than the other experiments.
To reproduce the paper results, we set the corresponding `timestamp` configuration variable in
[experiments/system_identification/optim_sysid.py](experiments/system_identification/optim_sysid.py).
For using the same model checkpoint as in the other experiments, run the code `with timestamp='latest'` (qualitatively,
the results are the same, but there is an additional outlier run for the force optimization).

#### Real-world experiment
To run the real-world experiment, place the file `real_world_data.pkl` from our [KEEPER repository](https://keeper.mpdl.mpg.de/d/c446493994b5486a843a/)
in the root project root folder and run
```bash
python experiments/trajectory_fitting/optim_pointcloud_real.py
```

### Rendering results for completed runs
Renderings of the trajectories in the experiments can be generated by running
```bash
python experiments/render_results.py <path/to/run> <render/output/path>
```

## License
See [LICENSE](LICENSE) and [NOTICE](NOTICE).

For license information on 3rd-party software we use in this project, see [NOTICE](NOTICE).
To comply with the Apache License, Version 2.0, any Derivative Works that You distribute must include a readable copy of
the attribution notices contained within the NOTICE file (see [LICENSE](LICENSE)).

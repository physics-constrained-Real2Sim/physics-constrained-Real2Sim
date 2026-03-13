# Training IGR

We provide instructions on how we preprocessed data for training IGR.

For the "bob and spot" shape space, we adapted the [preprocessing script](https://github.com/amosgropp/IGR/blob/master/code/preprocess/dfaust.py) provided for the dfaust to generate surface points and normals for bob and spot, which we downloaded from https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/.

For the ShapeNetV1 objects, we want to avoid points from internal surfaces.
We thus adapted the rendering-based [SampleVisibleMeshSurface](https://github.com/facebookresearch/DeepSDF/blob/main/src/SampleVisibleMeshSurface.cpp) program from [DeepSDF](https://github.com/facebookresearch/DeepSDF) to output surface normals for sampling 250000 points and storing points as well as normals as npy files.

We then created a data loader based on [the one for dfaust](https://github.com/amosgropp/IGR/blob/master/code/datasets/dfaustdataset.py) to load the preprocessed data for training IGR.

We provide the training configurations used in our experiments in [IGR_data/train_configs](IGR_data/train_configs) and the splits containing the object ids we used in [IGR_data/splits](IGR_data/splits).

# DeepNanoDesign - training a bi-diorectional neural network for the design of nano-photonics structures

DeepNanoDesign is a software library for training deep neural networks for the design and retrieval of nano-photonic structures.

## Run an experiment
Training a network:
1. Set-up your experiment in `configuration.lua`.
2. Run experiment:
```
th doall.lua
```

Running Genetics Algorithm:
```
th geneticsAlgorithm.lua
```

## Models
You can choose between:
1) Training a bi-directional model that given two spectrums predicts a geometry and then predicts back the two spectrums of the predicted
geometry.
2) Training an inverse network (GPN) that only predicts a geometry.
3) Training a direct network (SPN) that given a geometry predicts two spectrums. 
4) Running Genetic Algorithm (GA) to design a geometry for a given spectra.

## Citation

If you find this code useful in your research, please consider citing: 
```
I. Malkiel, M. Mrejen, A. Nagler, U. Arieli, L. Wolf and H. Suchowski, "Plasmonic nanostructure design and characterization via Deep Learning", Light: Science & Applications 7 (1), 60


I. Malkiel, M. Mrejen, A. Nagler, U. Arieli, L. Wolf and H. Suchowski, "Deep learning for the design of nano-photonic structures", 2018 IEEE International Conference on Computational Photography (ICCP), Pittsburgh, PA, 2018, pp. 1-14.
```
You can find the papers here: 

https://www.nature.com/articles/s41377-018-0060-7

https://ieeexplore.ieee.org/document/8368462/

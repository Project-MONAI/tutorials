# AutoML

Here we showcase the most recent AutoML techniques in medical imaging based on MONAI modules.

## [DiNTS: Differentiable Neural Network Topology Search](./DiNTS)
Recently, neural architecture search (NAS) has been applied to automatically
search high-performance networks for medical image segmentation. The NAS search
space usually contains a network topology level (controlling connections among
cells with different spatial scales) and a cell level (operations within each
cell). Existing methods either require long searching time for large-scale 3D
image datasets, or are limited to pre-defined topologies (such as U-shaped or
single-path).

In this work, we focus on three important aspects of NAS in 3D medical image
segmentation: flexible multi-path network topology, high search efficiency, and
budgeted GPU memory usage. A novel differentiable search framework is proposed
to support fast gradient-based search within a highly flexible network topology
search space. The discretization of the searched optimal continuous model in
differentiable scheme may produce a sub-optimal final discrete model
(discretization gap). Therefore, we propose a topology loss to alleviate this
problem. In addition, the GPU memory usage for the searched 3D model is limited
with budget constraints during search. The Differentiable Network Topology
Search scheme (DiNTS) was evaluated on the Medical Segmentation Decathlon (MSD)
challenge with state-of-the-art performance.

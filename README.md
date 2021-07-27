# Neural Weighted A*

Can we learn to navigate a terrain effectively by just looking at its map? Our [paper](https://arxiv.org/abs/2105.01480) describes Neural Weighted A*, a deep learning architecture that accurately predicts, from the image of the navigation area, the costs of traversing local regions and a global heuristic function for reaching the destination. This repository contains the source code of the experiments involved in the paper.

## Table of contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Datasets](#datasets)

## Introduction

![teaser](/utils/NWAStar_teaser.jpg)

[Neural Weighted A*](https://arxiv.org/abs/2105.01480) is a differentiable planner able to learn graph costs and heuristic functions from planning examples. It arises from the recent trend of incorporating differentiable combinatorial solvers into deep learning pipelines to ease the learning procedure for combinatorial data. Most recent works are able to learn either cost functions or heuristic functions, but not both. To the best of our knowledge, Neural Weighted A* is the first architecture able to learn graph costs and heuristic functions, aware of the [admissibility constraint](https://en.wikipedia.org/wiki/Admissible_heuristic). Training occurs end-to-end on raw images with direct supervision on planning examples, thanks to a differentiable A* solver integrated into the architecture, built upon [black-box differentiation](https://arxiv.org/abs/1912.02175) and [Neural A*](https://arxiv.org/abs/2009.07476).

## Requirements

* torch
* numpy
* ray
* cv2
* scipy
* yapt (https://github.com/marcociccone/yapt)

## Datasets

The experiments involved two tile-based navigation datasets, available [here](https://github.com/archettialberto/tilebased_navigation_datasets).

![datasets](/utils/NWAStar_datasets.jpg)

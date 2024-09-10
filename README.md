# Optimal Power Flow Prediction Using Graph Neural Networks

## Overview

This project aims to predict the active power outputs of power generators using Graph Neural Networks (GNNs). The objective is to develop a model that leverages graph-based data to accurately predict generator power outputs based on the state of the power system.

## Source Paper & Dataset

The datasets used in this project are based on the paper [**"OPFData: Large-scale datasets for AC optimal power flow with topological perturbations"**](https://arxiv.org/pdf/2406.07234). This paper provides comprehensive datasets for AC Optimal Power Flow (OPF) problems, including data with various topological perturbations. These datasets are crucial for testing and evaluating the robustness and performance of power system models.

For implementation and access to these datasets within PyTorch Geometric, please refer to the [PyTorch Geometric documentation on OPFDataset](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/opf.html).


## Data Handling with PyTorch Geometric

We use [**PyTorch Geometric**](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html) for handling and processing graph data. PyTorch Geometric is a library specifically designed for working with graph-structured data and is well-suited for this project due to its efficient and flexible tools.

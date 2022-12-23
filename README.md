# Neural Network from Scratch

[![CI](https://github.com/mrahhal/neural-network-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/mrahhal/neural-network-from-scratch/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Neural network implementations from scratch in Rust.

## Setup & Run

Dataset used is [mnist](http://yann.lecun.com/exdb/mnist/). Download the 4 archives and extract them into "datasets/mnist" folder.

A `cargo run` will setup the network, train it on a subset of the data while testing the result after each epoch infinitely until the target accuracy is reached. Currently, params resulting after training is not cached. Also, for now random seeds are used to produce reproducable and consistent results.

This is running on the CPU right now, so it's not very fast. Ideally would want to make use of GPU computation.

## Current implementations

- [MLP](./src/neural_networks/mlp.rs): A multi-layer perceptron neural network.

## Ideas

Upcoming implementations:
- CNN
- RNN (LSTM)
- Transformer

Improvements:
- Serialize/Deserialize params
- Split lib crates
- GPU

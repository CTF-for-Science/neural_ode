# NeuralODE

This directory contains an implementation of the NeuralODE model for the CTF for Science Framework, in pytrorch and using diffeq library. NeuralODE is a deep learning architecture designed for learning differentil equations

## Table of Contents
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Output](#output)
- [References](#references)

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -U pip
pip install torch
pip install torchdiffeq
cd ../..
pip install -e .[all]
```

## Usage

### Running the Model

To run the model, use the `run.py` script from the model directory:

```bash
cd models/neural_ode
python run.py config/config_KS_batch_all.yaml
python run.py config/config_Lorenz_batch_all.yaml
```

### Hyperparameter Tuning

To run hyperparameter tuning:

```bash
cd models/neural_ode
python optimize_parameters.py tuning_config/config_KS.yaml
python optimize_parameters.py tuning_config/config_Lorenz.yaml
```

## Configuration

Configuration files are located in the `config/` directory:
- `config_KS_batch_all.yaml`: Runs the FNO model on `PDE_KS`
- `config_Lorenz_batch_all.yaml`: Runs the FNO model on `ODE_Lorenz`

Each configuration file contains:
- Dataset specifications
- Model hyperparameters
- Training parameters

## Dependencies

The NeuralODE implementation requires the following dependencies:
- PyTorch (>= 1.8.0, < 2.0.0)
- NumPy (>= 1.19.0, < 2.0.0)
- PyYAML (>= 5.1.0, < 6.0.0)
- torchdiffeq (>=0.2.5)
- ctf4science python project

## Model Architecture
**Neural differential equations**  are a class of models in  [machine learning](https://en.wikipedia.org/wiki/Machine_learning "Machine learning")  that combine  [neural networks](https://en.wikipedia.org/wiki/Neural_networks "Neural networks")  with the mathematical framework of  [differential equations](https://en.wikipedia.org/wiki/Differential_equations "Differential equations").[[1]](https://en.wikipedia.org/wiki/Neural_differential_equation#cite_note-:06-1)  These models provide an alternative approach to neural network design, particularly for systems that evolve over time or through continuous transformations.

The most common type, a  **neural ordinary differential equation (neural ODE)**, defines the evolution of a system's state using an  [ordinary differential equation](https://en.wikipedia.org/wiki/Ordinary_differential_equation "Ordinary differential equation")  whose dynamics are governed by a neural network.[[2]](https://en.wikipedia.org/wiki/Neural_differential_equation) 

For further description of the architecture, see https://en.wikipedia.org/wiki/Neural_differential_equation

    


  
## Output

The model generates several types of outputs:

### Training Outputs from run.py
- Predictions for each sub-dataset
- Evaluation metrics (saved in YAML format)
- Batch results summary
- Location: `results/` directory under a unique batch identifier

### Tuning Outputs from optimize_parameters.py
- Optimal hyperparameters
- Tuning history
- Performance metrics
- Location: `results/tune_result` directory

## References

-   Chen, T. Q., Rubanova, Y., Bettencourt, J., Duvenaud, D. (2018). _Neural Ordinary Differential Equations_. NeurIPS.

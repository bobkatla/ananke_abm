# Ananke ABM

**Agent-Based Model for synthetic population data and activity predictions**

Ananke ABM is a Python package designed to connect synthetic population data with activity predictions for agent-based modeling (ABM). It aims to use Graph Neural Networks (especially Graph Attention Models) with ODE-inspired approach to solve the issue as a spatial-temporal issue.

## Features

- 🤖 **Machine Learning Models**: Advanced ML models including Graph Neural Networks (GNN) for population and activity modeling
- 📊 **Data Generation**: Tools for generating synthetic population data
- 🔍 **Model Inference**: Comprehensive inference capabilities for trained models, was build with Melbourne data (VISTA) in mind

## Installation

### Prerequisites

- Python >= 3.10
- [UV package manager](https://github.com/astral-sh/uv) (recommended)

### Install with UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/bobkatla/ananke_abm.git
cd ananke_abm

# Install the package in development mode
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"
```

### Install with pip

```bash
# Clone the repository
git clone https://github.com/bobkatla/ananke_abm.git
cd ananke_abm

# Install the package
pip install -e .
```

## Quick Start

### Basic Usage
TODO: in working still

```python
import ananke_abm

# Import specific modules
from ananke_abm import models, utils, data_generator

# Access model submodules
from ananke_abm.models import inference, run, gnn_embed

```

### Command Line Interface

The package provides a command-line interface:
TODO: in working

```bash
# Get version and basic info
ananke --version
ananke --help
```

## Package Structure

```
src/ananke_abm/
├── __init__.py              # Main package initialization
├── models/                  # Machine learning models
│   ├── inference/          # Model inference functionality
│   ├── run/                # Model execution utilities  
│   └── gnn_embed/          # Graph Neural Network embeddings
├── utils/                   # Utility functions and helpers
└── data_generator/          # Synthetic data generation tools
```

## Development

### Setting up Development Environment

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ananke_abm
   ```

2. **Install UV** (if not already installed):
   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Install in development mode**:
   ```bash
   uv pip install -e ".[dev]"
   ```
   
## Research Context

This package is developed as part of Duc Minh (Bob) La's PhD research at Monash University focusing on:
- Synthetic population generation
- Activity-based modeling
- Agent-based modeling applications
- Graph neural networks for population modeling


## Citation

If you use this software in your research, please cite:
TODO: in working

## Contact

**Bob La**  
Email: duc.la@monash.edu  
Institution: Monash University

*This project is part of ongoing PhD research at Monash University.*

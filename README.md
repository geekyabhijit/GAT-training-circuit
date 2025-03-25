# Graph Attention Network (GAT) for Circuit Topology Classification

This repository contains an implementation of a Graph Attention Network (GAT) with multihead attention for analog circuit topology classification tasks.

## Circuit Topologies
The model is designed to classify the following 8 analog circuit topologies:

1. Simple Current Mirror
2. Cascaded Current Mirror
3. Differential Pair
4. Single-Stage Differential Amplifier
5. Two-Stage Amplifier
6. LDO (Low Dropout Regulator)
7. Bandgap Reference
8. Comparator

## Model Architecture

1. **Input Layer**:
   - Input: Node features of size F = 4 (one-hot encoding for device types: NMOS, PMOS, Resistor, Capacitor)
   - Graph: Variable-sized circuit graphs with different connectivity patterns

2. **GAT Layer 1**:
   - Multi-head Attention: 8 heads, output dimension per head: 8
   - Activation: ELU (Exponential Linear Unit)
   - Dropout: 0.6

3. **GAT Layer 2**:
   - Multi-head Attention: 1 head, output dimension: 8 classes (one for each circuit topology)
   - Activation: Softmax for classification

## Hyperparameters

- **Optimizer**: Adam
- **Learning Rate**: 0.005
- **Epochs**: 200
- **Loss Function**: Cross-Entropy Loss
- **Dropout**: 0.6
- **Weight Decay (L2)**: 0.0005
- **Evaluation Metrics**: Accuracy, F1-score, Recall, Confusion Matrix

## Data Representation

- **Nodes**: Represent electronic components (transistors, resistors, capacitors)
- **Edges**: Represent connections between components
- **Node Features**: 4-dimensional one-hot vectors indicating device types:
  - [1,0,0,0] = NMOS transistor
  - [0,1,0,0] = PMOS transistor
  - [0,0,1,0] = Resistor
  - [0,0,0,1] = Capacitor
- **Graph Labels**: Indicate which circuit topology the graph represents

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

   > **Note**: PyTorch Geometric requires additional dependencies. If you encounter issues, refer to the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Usage

### Training with Synthetic Data

To train with synthetic data (useful when you don't have real circuit netlists):

```bash
python gat_training.py
```

This will:
- Generate synthetic data for each circuit topology
- Train the GAT model with the specified hyperparameters
- Evaluate the model performance using accuracy, F1-score, recall and confusion matrix
- Save the best model to `best_circuit_gat_model.pth`
- Generate detailed training metrics and visualization in `circuit_training_metrics.png`
- Print per-class performance metrics for each circuit topology

### Training with Real SPICE Netlists

To train with real circuit netlists (SPICE format):

1. Organize your netlists in the following directory structure:
```
data/
├── simple_current_mirror/
│   ├── circuit1.sp
│   └── ...
├── differential_pair/
│   ├── circuit1.sp
│   └── ...
└── ...
```

2. Run the netlist-based training:
```bash
python train_with_netlists.py
```

For a quick start with example netlists:
```bash
python copy_examples.py  # Copies example netlists to data directory
python train_with_netlists.py
```

See [README_NETLIST.md](README_NETLIST.md) for detailed information on using netlists.

## Netlist Parser

The repository includes a SPICE netlist parser (`netlist_to_graph.py`) that:

1. Extracts components from standard SPICE netlists
2. Identifies component types (NMOS, PMOS, resistors, capacitors)
3. Creates graph representations of circuit connectivity
4. Converts netlists into PyTorch Geometric's Data format

The parser recognizes:
- MOSFETs (`M` prefix) - distinguished as NMOS or PMOS based on model parameters
- Resistors (`R` prefix)
- Capacitors (`C` prefix)
- Net connections and shared nodes

## Model Evaluation

The training process includes:
- Training/validation/test split (70%/15%/15%)
- Per-epoch evaluation of accuracy, F1-score, and recall
- Confusion matrix visualization
- Per-class performance metrics
- Automatic model saving (best validation accuracy)

## Customization

You can modify:
- Hyperparameters in the training scripts
- Model architecture (e.g., layer dimensions, attention heads)
- Netlist parser to support additional component types
- Circuit topology characteristics in the synthetic data generator

For real-world applications, replace the synthetic data with your actual circuit netlists. 
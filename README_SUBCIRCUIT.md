# Detecting Subcircuits in Complex Circuit Netlists

This document explains how to use the subcircuit detection functionality to identify common circuit topologies within larger, more complex circuit designs.

## Overview

The subcircuit detector uses a pre-trained GAT model to identify potential subcircuits within a larger circuit by:

1. Breaking down the circuit graph into overlapping subgraphs
2. Classifying each subgraph using the trained model
3. Filtering and prioritizing detected subcircuits
4. Visualizing the results

## Prerequisites

Before using the subcircuit detector, you need:

1. A trained GAT model (created using `train_with_netlists.py` or `gat_training.py`)
2. A complex SPICE netlist containing multiple subcircuits

## Usage

### Basic Command

```bash
python detect_subcircuits.py --netlist examples/complex_circuit.sp
```

### Options

- `--netlist`: Path to the input SPICE netlist (required)
- `--model`: Path to the trained model file (default: 'best_circuit_gat_model.pth')
- `--output`: Path for the output visualization image (default: 'detected_subcircuits.png')
- `--confidence`: Confidence threshold for detection (default: 0.7, range: 0-1)
- `--radius`: Neighborhood radius for subcircuit extraction (default: 2)

### Example

```bash
python detect_subcircuits.py --netlist examples/complex_circuit.sp --confidence 0.6 --radius 3
```

## How It Works

The subcircuit detector operates in several steps:

1. **Graph Creation**: Converts the SPICE netlist into a graph representation
2. **Subgraph Extraction**: For each component node:
   - Extracts a subgraph containing the node and its neighbors up to a specified radius
   - Components are identified by their one-hot encoded features (NMOS, PMOS, Resistor, Capacitor)
3. **Classification**: Applies the GAT model to classify each subgraph
4. **Filtering**: Removes overlapping or low-confidence detections
5. **Visualization**: Generates a graph visualization with highlighted subcircuits

## Interpretation

The detector outputs:

1. Terminal text showing:
   - List of detected subcircuits
   - Confidence score for each detection
   - Components involved in each subcircuit
   
2. Visualization image showing:
   - Complete circuit graph with component colors (NMOS, PMOS, Resistor, Capacitor)
   - Highlighted regions indicating detected subcircuits
   - Labels showing topology type and confidence

## Limitations

- Detection accuracy depends on the quality of the trained model
- Performance varies based on the similarity between training data and the target circuit
- Overlapping subcircuits may not be perfectly separated
- Very complex circuits might require adjusting the confidence threshold and radius

## Tips for Better Results

1. **Adjust confidence threshold**:
   - Lower threshold (e.g., 0.5) to detect more potential subcircuits
   - Higher threshold (e.g., 0.8) for higher precision but fewer detections

2. **Adjust neighborhood radius**:
   - Smaller radius for detecting compact subcircuits
   - Larger radius for detecting larger topologies

3. **Improve the base model**:
   - Train with more examples of each topology
   - Include variations of circuits in training data
   - Use netlists with similar formatting to your target circuit

## Example Expected Output

```
Using device: cpu
Loading model from best_circuit_gat_model.pth...
Processing netlist examples/complex_circuit.sp...
Circuit graph created with 28 nodes and 42 edges.
Detecting subcircuits with confidence threshold 0.7...
Detected 3 potential subcircuits:
  1. Simple Current Mirror (confidence: 0.89)
     Size: 5 components
     Center component: MCM1
     Components: ['MCM1', 'MCM2', 'IREF', 'VSS', 'CM_DRAIN_REF']
  2. Differential Pair (confidence: 0.85)
     Size: 7 components
     Center component: MDP1
     Components: ['MDP1', 'MDP2', 'RDP1', 'RDP2', 'VDD']...
  3. Cascaded Current Mirror (confidence: 0.78)
     Size: 6 components
     Center component: MCC1
     Components: ['MCC1', 'MCC2', 'MCC3', 'RLOAD_CC', 'ICC', 'CC_DRAIN_REF']
Generating visualization in detected_subcircuits.png...
Visualization saved to detected_subcircuits.png
```

## Integration with Other Tools

The subcircuit detector can be integrated with other tools in this repository:

- Use after training a model with `train_with_netlists.py`
- Combine with the netlist parser for custom preprocessing
- Extend with additional analysis for specific circuit types 
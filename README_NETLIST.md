# Training GAT Models with Circuit Netlists

This guide explains how to train the Graph Attention Network (GAT) model using real SPICE netlists of analog circuits instead of synthetic data.

## Directory Structure

The system expects netlists to be organized in the following directory structure:

```
data/
├── simple_current_mirror/
│   ├── circuit1.sp
│   ├── circuit2.sp
│   └── ...
├── cascaded_current_mirror/
│   ├── circuit1.sp
│   └── ...
├── differential_pair/
│   └── ...
└── ...
```

Each subdirectory in the `data/` folder should contain SPICE netlist files (`.sp`, `.spice`, or `.cir` extensions) for a specific circuit topology.

## Netlist Format

The system supports standard SPICE netlists. Here's an example of a Simple Current Mirror netlist:

```spice
* Simple Current Mirror SPICE Netlist
.include /path/to/models/mosfet.lib

* Power supply
VDD VDD 0 DC 1.8V

* Current mirror circuit
* Reference current
IREF VDD DRAIN_REF DC 100u
* Diode-connected reference transistor
M1 DRAIN_REF DRAIN_REF 0 0 NMOS W=10u L=1u m=1
* Mirror transistor 
M2 OUT DRAIN_REF 0 0 NMOS W=10u L=1u m=1
* Load resistor
RLOAD VDD OUT 1k

* Analysis commands
.op
.DC IREF 50u 150u 10u

.end
```

The netlist parser recognizes:
- MOSFETs (M prefix): Classified as NMOS (default) or PMOS (based on model name)
- Resistors (R prefix)
- Capacitors (C prefix)
- Other components are also extracted for structural information

## Training with Netlists

1. Prepare your netlists:
   - Collect SPICE netlists for each circuit topology
   - Organize them in the directory structure shown above
   - Ensure netlists use standard SPICE syntax

2. Run the training script:
   ```
   python train_with_netlists.py
   ```

3. The script will:
   - Load all netlists from the data directories
   - Convert them into graph representations
   - Split the dataset into training (70%), validation (15%), and test (15%) sets
   - Train the GAT model according to specified hyperparameters
   - Evaluate the model's performance
   - Save the best model and generate performance visualizations

## How Netlists Are Converted to Graphs

1. **Node Extraction**:
   - Component instances (MOSFETs, resistors, capacitors) are extracted as nodes
   - Connection points are identified
   
2. **Feature Encoding**:
   - Each node is assigned a 4-dimensional one-hot encoded feature vector
   - [1,0,0,0] = NMOS
   - [0,1,0,0] = PMOS
   - [0,0,1,0] = Resistor
   - [0,0,0,1] = Capacitor

3. **Edge Creation**:
   - Edges are created between components that share connection points
   - This forms a graph representing the circuit's connectivity

4. **Label Assignment**:
   - All nodes in a graph are assigned the same label based on the parent directory name
   - This label represents the circuit topology class

## Tips for Better Results

1. **Consistent Netlist Format**:
   - Use consistent naming conventions in your netlists
   - Ensure all components have appropriate prefixes (M for MOSFETs, R for resistors, etc.)
   
2. **Diverse Examples**:
   - Include multiple variants of each circuit topology with different component values
   - Include circuits from different design styles and applications
   
3. **Clean Netlists**:
   - Remove unnecessary components or subcircuits that aren't relevant to topology identification
   - Consider preprocessing netlists to standardize formats

4. **Model Tuning**:
   - Adjust hyperparameters like learning rate, hidden dimensions, and dropout based on your dataset size
   - Increase epochs for larger datasets

## Customizing the Parser

If your netlists contain specialized components or follow non-standard formats, you may need to modify the `netlist_to_graph.py` file to accommodate these differences. 
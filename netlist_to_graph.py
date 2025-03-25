import os
import re
import torch
from torch_geometric.data import Data, Dataset
import numpy as np

# Component type mapping
COMPONENT_TYPES = {
    'M': 0,  # MOSFET (0 = NMOS, 1 = PMOS will be determined by model parameter)
    'R': 2,  # Resistor
    'C': 3,  # Capacitor
    # Add more component types as needed
}

def parse_netlist(netlist_path):
    """
    Parse a SPICE netlist file into nodes and edges.
    
    Args:
        netlist_path: Path to the SPICE netlist file
        
    Returns:
        nodes: Dictionary mapping node names to indices and component types
        edges: List of edge tuples (source, target)
    """
    nodes = {}  # Maps node name to (index, component_type)
    edges = []
    node_counter = 0
    
    with open(netlist_path, 'r') as f:
        lines = f.readlines()
    
    # First pass: identify all nodes and their types
    for line in lines:
        line = line.strip()
        if line.startswith('*') or line.startswith('.') or not line:
            continue  # Skip comments and control statements
            
        parts = line.split()
        if not parts:
            continue
            
        component_name = parts[0]
        component_type = component_name[0].upper()  # First letter of component name (M, R, C, etc)
        
        if component_type not in COMPONENT_TYPES:
            continue  # Skip unknown component types
            
        # Extract connections (pins)
        connections = []
        for i in range(1, len(parts)):
            if parts[i].startswith('+') or parts[i].startswith('-') or parts[i].startswith('='):
                break  # Stop at parameters section
            connections.append(parts[i])
        
        # Add component node
        if component_name not in nodes:
            # For MOSFETs, check model type (NMOS or PMOS)
            is_pmos = False
            if component_type == 'M':
                for part in parts:
                    if part.startswith('model=') or part.startswith('MODEL='):
                        model = part.split('=')[1].upper()
                        is_pmos = 'PMOS' in model or 'PFET' in model
                        break
            
            comp_type_idx = COMPONENT_TYPES[component_type]
            # For MOSFETs, adjust index based on NMOS/PMOS
            if component_type == 'M' and is_pmos:
                comp_type_idx = 1  # PMOS
                
            nodes[component_name] = (node_counter, comp_type_idx)
            node_counter += 1
        
        # Add connection edges
        for i in range(len(connections)):
            for j in range(i+1, len(connections)):
                # Create edges between connection points
                if connections[i] != '0' and connections[j] != '0':  # Skip ground connections
                    # Add net nodes if they don't exist
                    if connections[i] not in nodes:
                        nodes[connections[i]] = (node_counter, -1)  # -1 for net nodes
                        node_counter += 1
                    if connections[j] not in nodes:
                        nodes[connections[j]] = (node_counter, -1)  # -1 for net nodes
                        node_counter += 1
                    
                    edges.append((nodes[connections[i]][0], nodes[connections[j]][0]))
                    edges.append((nodes[connections[j]][0], nodes[connections[i]][0]))  # Bidirectional
    
    return nodes, edges

def netlist_to_data(netlist_path, topology_label):
    """
    Convert a SPICE netlist to a PyTorch Geometric Data object.
    
    Args:
        netlist_path: Path to the SPICE netlist file
        topology_label: Integer label for the circuit topology
        
    Returns:
        data: PyTorch Geometric Data object
    """
    nodes, edges = parse_netlist(netlist_path)
    
    # Create node feature matrix (one-hot encoding)
    x = torch.zeros((len(nodes), 4))  # 4 types: NMOS, PMOS, Resistor, Capacitor
    
    for node_name, (node_idx, node_type) in nodes.items():
        if node_type >= 0:  # Skip net nodes (type = -1)
            x[node_idx, node_type] = 1.0
    
    # Create edge index tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Create node labels (all nodes have the same label - the circuit topology)
    y = torch.tensor([topology_label] * len(nodes), dtype=torch.long)
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data

class NetlistDataset(Dataset):
    """
    Dataset class for loading multiple netlists.
    """
    def __init__(self, root, netlist_dirs, transform=None, pre_transform=None):
        """
        Args:
            root: Root directory for the dataset
            netlist_dirs: Dictionary mapping topology labels to directories containing netlists
        """
        self.netlist_dirs = netlist_dirs
        self.topology_labels = {dir_name: label for label, dir_name in enumerate(netlist_dirs)}
        self.data_list = []
        
        # Process all netlists
        for dir_name, dir_path in netlist_dirs.items():
            label = self.topology_labels[dir_name]
            for filename in os.listdir(dir_path):
                if filename.endswith('.sp') or filename.endswith('.spice') or filename.endswith('.cir'):
                    netlist_path = os.path.join(dir_path, filename)
                    try:
                        data = netlist_to_data(netlist_path, label)
                        self.data_list.append(data)
                    except Exception as e:
                        print(f"Error processing {netlist_path}: {e}")
        
        super(NetlistDataset, self).__init__(root, transform, pre_transform)
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]
    
    def process(self):
        pass
    
    def _download(self):
        pass

def load_netlist_dataset(root_dir, data_dirs):
    """
    Load netlists from directories into a dataset.
    
    Args:
        root_dir: Root directory for the dataset
        data_dirs: Dictionary mapping topology names to directories containing netlists
        
    Returns:
        dataset: NetlistDataset object
    """
    return NetlistDataset(root_dir, data_dirs)

# Example usage
if __name__ == "__main__":
    # Example directory structure:
    # data/
    #   - simple_current_mirror/
    #       - circuit1.sp
    #       - circuit2.sp
    #   - differential_pair/
    #       - circuit1.sp
    #       - circuit2.sp
    
    data_dirs = {
        "simple_current_mirror": "data/simple_current_mirror",
        "cascaded_current_mirror": "data/cascaded_current_mirror",
        "differential_pair": "data/differential_pair",
        "single_stage_differential_amplifier": "data/single_stage_differential_amplifier",
        "two_stage_amplifier": "data/two_stage_amplifier",
        "ldo": "data/ldo",
        "bandgap_reference": "data/bandgap_reference",
        "comparator": "data/comparator"
    }
    
    dataset = load_netlist_dataset("data", data_dirs)
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Example how to access data
    if len(dataset) > 0:
        data = dataset[0]
        print(f"Node features shape: {data.x.shape}")
        print(f"Edge index shape: {data.edge_index.shape}")
        print(f"Labels shape: {data.y.shape}") 
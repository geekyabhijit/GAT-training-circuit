import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from netlist_to_graph import parse_netlist, COMPONENT_TYPES
import os
import copy
from collections import defaultdict

class SubcircuitDetector:
    """
    Detects potential subcircuit topologies within a larger circuit using a pre-trained GAT model.
    Uses a sliding window approach on the graph structure to identify subcircuits.
    """
    
    def __init__(self, model, device, circuit_topologies):
        """
        Initialize the detector with a pre-trained model.
        
        Args:
            model: Pre-trained GAT model for topology classification
            device: Device to run inference on (CPU or CUDA)
            circuit_topologies: List of topology names corresponding to model outputs
        """
        self.model = model
        self.device = device
        self.circuit_topologies = circuit_topologies
        self.model.eval()  # Set model to evaluation mode
        
    def _extract_subgraph(self, data, center_node, radius=2):
        """
        Extract a subgraph centered at a node with a given radius.
        
        Args:
            data: PyTorch Geometric Data object of the full circuit
            center_node: Index of the center node
            radius: Neighborhood radius (number of hops)
            
        Returns:
            subgraph: PyTorch Geometric Data object of the extracted subgraph
        """
        # Convert to networkx for easier subgraph extraction
        G = to_networkx(data, to_undirected=True)
        
        # Get nodes within radius
        subgraph_nodes = set([center_node])
        frontier = {center_node}
        
        for _ in range(radius):
            new_frontier = set()
            for node in frontier:
                neighbors = set(G.neighbors(node))
                new_frontier.update(neighbors - subgraph_nodes)
            subgraph_nodes.update(new_frontier)
            frontier = new_frontier
            if not frontier:
                break
        
        # Create subgraph
        subgraph_nodes = sorted(list(subgraph_nodes))
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph_nodes)}
        
        # Extract features for subgraph nodes
        x = data.x[subgraph_nodes]
        
        # Extract edges that connect nodes in the subgraph
        edge_index = []
        for i, j in data.edge_index.t().tolist():
            if i in subgraph_nodes and j in subgraph_nodes:
                edge_index.append([node_mapping[i], node_mapping[j]])
        
        if not edge_index:
            return None  # No edges in subgraph
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create new Data object for subgraph
        subgraph = Data(x=x, edge_index=edge_index)
        
        # Store original node indices for reference
        subgraph.original_indices = torch.tensor(subgraph_nodes, dtype=torch.long)
        
        return subgraph
    
    def detect_subcircuits(self, data, confidence_threshold=0.7):
        """
        Detect potential subcircuits in the larger circuit.
        
        Args:
            data: PyTorch Geometric Data object of the full circuit
            confidence_threshold: Minimum confidence score to consider a match
            
        Returns:
            detected: List of dictionaries containing detected subcircuits
        """
        if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
            return []
            
        detected = []
        
        # For each node, extract a subgraph and classify it
        for center_node in range(len(data.x)):
            # Skip nodes that don't represent actual components (e.g., net nodes)
            if data.x[center_node].sum() == 0:
                continue
                
            # Extract subgraph centered at this node
            subgraph = self._extract_subgraph(data, center_node)
            if subgraph is None or len(subgraph.x) < 3:  # Skip if too small
                continue
            
            # Run inference
            subgraph = subgraph.to(self.device)
            with torch.no_grad():
                out = self.model(subgraph.x, subgraph.edge_index)
                
            # Get predictions for all nodes
            probs = F.softmax(out, dim=1)
            avg_probs = probs.mean(dim=0)  # Average across nodes
            
            # Get the most likely topology and its confidence
            max_prob, predicted = avg_probs.max(dim=0)
            max_prob = max_prob.item()
            predicted = predicted.item()
            
            # If confidence is high enough, consider it a match
            if max_prob >= confidence_threshold:
                detected.append({
                    'center_node': center_node,
                    'topology': self.circuit_topologies[predicted],
                    'confidence': max_prob,
                    'nodes': subgraph.original_indices.cpu().numpy().tolist(),
                    'size': len(subgraph.x)
                })
        
        # Post-process: remove overlapping subcircuits with lower confidence
        return self._remove_overlaps(detected)
    
    def _remove_overlaps(self, detected):
        """
        Remove overlapping subcircuit detections by keeping the highest confidence ones.
        
        Args:
            detected: List of detected subcircuits
            
        Returns:
            filtered: List of filtered subcircuits with overlaps removed
        """
        if not detected:
            return []
            
        # Sort by confidence (descending)
        sorted_detected = sorted(detected, key=lambda x: x['confidence'], reverse=True)
        
        # Keep track of which nodes are already assigned to a subcircuit
        assigned_nodes = set()
        filtered = []
        
        for detection in sorted_detected:
            nodes = set(detection['nodes'])
            
            # Calculate overlap with already assigned nodes
            overlap = nodes.intersection(assigned_nodes)
            overlap_ratio = len(overlap) / len(nodes) if nodes else 0
            
            # If overlap is small enough, keep this detection
            if overlap_ratio < 0.3:  # Allow some overlap
                filtered.append(detection)
                assigned_nodes.update(nodes)
        
        return filtered
        
    def visualize_detection(self, data, detected, save_path=None):
        """
        Visualize the circuit graph with detected subcircuits highlighted.
        
        Args:
            data: PyTorch Geometric Data object
            detected: List of detected subcircuits
            save_path: Path to save the visualization, or None to display
        """
        G = to_networkx(data, to_undirected=True)
        
        # Create position layout
        pos = nx.spring_layout(G, seed=42)
        
        plt.figure(figsize=(12, 10))
        
        # Draw the full graph
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        
        # Prepare node colors based on component types
        component_colors = []
        for i in range(len(data.x)):
            if torch.argmax(data.x[i]).item() == 0:
                component_colors.append('skyblue')  # NMOS
            elif torch.argmax(data.x[i]).item() == 1:
                component_colors.append('pink')     # PMOS
            elif torch.argmax(data.x[i]).item() == 2:
                component_colors.append('lightgreen')  # Resistor
            elif torch.argmax(data.x[i]).item() == 3:
                component_colors.append('yellow')   # Capacitor
            else:
                component_colors.append('gray')     # Other/Net
        
        nx.draw_networkx_nodes(G, pos, node_color=component_colors, alpha=0.5)
        
        # Draw node labels (component indices)
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        # Highlight detected subcircuits
        for i, detection in enumerate(detected):
            subgraph_nodes = detection['nodes']
            
            # Draw convex hull or circle around the subcircuit
            node_positions = np.array([pos[node] for node in subgraph_nodes])
            
            # Calculate the centroid of subcircuit nodes
            centroid = node_positions.mean(axis=0)
            
            # Draw a circle around the subcircuit
            radius = max([np.linalg.norm(pos[node] - centroid) for node in subgraph_nodes]) + 0.05
            circle = plt.Circle(centroid, radius, fill=False, edgecolor=f'C{i}', linewidth=2, alpha=0.7)
            plt.gca().add_patch(circle)
            
            # Add label for the detected topology
            label = f"{detection['topology']} ({detection['confidence']:.2f})"
            plt.text(centroid[0], centroid[1] + radius + 0.05, label,
                    horizontalalignment='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        plt.title('Circuit Graph with Detected Subcircuits')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def preprocess_netlist(self, netlist_path):
        """
        Process a SPICE netlist file to create a graph representation.
        
        Args:
            netlist_path: Path to the SPICE netlist file
            
        Returns:
            data: PyTorch Geometric Data object
        """
        # Parse netlist
        nodes, edges = parse_netlist(netlist_path)
        
        # Create node feature matrix (one-hot encoding)
        x = torch.zeros((len(nodes), 4))  # 4 types: NMOS, PMOS, Resistor, Capacitor
        
        for node_name, (node_idx, node_type) in nodes.items():
            if node_type >= 0:  # Skip net nodes (type = -1)
                x[node_idx, node_type] = 1.0
        
        # Create edge index tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index)
        
        # Store component names for reference
        data.component_names = []
        for node_name, (node_idx, _) in sorted(nodes.items(), key=lambda x: x[1][0]):
            data.component_names.append(node_name)
        
        return data


def load_model(model_path, model_class, model_params, device):
    """
    Load a trained GAT model from disk.
    
    Args:
        model_path: Path to the saved model
        model_class: GAT model class
        model_params: Dictionary of parameters for model initialization
        device: Device to load the model onto
        
    Returns:
        model: Loaded model
    """
    model = model_class(**model_params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


if __name__ == "__main__":
    print("Subcircuit Detector module. Import and use in other scripts.") 
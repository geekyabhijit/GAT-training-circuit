#!/usr/bin/env python3
import torch
import argparse
import os
from gat_training import GATModel
from subcircuit_detector import SubcircuitDetector, load_model

# Define the circuit topologies
CIRCUIT_TOPOLOGIES = [
    "Simple Current Mirror",
    "Cascaded Current Mirror",
    "Differential Pair",
    "Single-Stage Differential Amplifier",
    "Two-Stage Amplifier",
    "LDO",
    "Bandgap Reference",
    "Comparator"
]

def main():
    parser = argparse.ArgumentParser(description='Detect subcircuits in a SPICE netlist using a trained GAT model.')
    parser.add_argument('--netlist', type=str, required=True, help='Path to input SPICE netlist')
    parser.add_argument('--model', type=str, default='best_circuit_gat_model.pth', help='Path to trained model')
    parser.add_argument('--output', type=str, default='detected_subcircuits.png', help='Output image path')
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold (0-1)')
    parser.add_argument('--radius', type=int, default=2, help='Neighborhood radius for subcircuit extraction')
    args = parser.parse_args()
    
    # Check if the input file exists
    if not os.path.exists(args.netlist):
        print(f"Error: Input netlist file '{args.netlist}' not found.")
        return
    
    # Check if the model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        return
    
    # Set device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define model parameters (must match the saved model)
    model_params = {
        'input_dim': 4,
        'hidden_dim': 8,
        'output_dim': 8,
        'heads1': 8,
        'heads2': 1,
        'dropout': 0.6
    }
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, GATModel, model_params, device)
    
    # Create the subcircuit detector
    detector = SubcircuitDetector(model, device, CIRCUIT_TOPOLOGIES)
    
    # Preprocess the netlist
    print(f"Processing netlist {args.netlist}...")
    data = detector.preprocess_netlist(args.netlist)
    print(f"Circuit graph created with {len(data.x)} nodes and {len(data.edge_index[0])} edges.")
    
    # Detect subcircuits
    print(f"Detecting subcircuits with confidence threshold {args.confidence}...")
    detected = detector.detect_subcircuits(data, confidence_threshold=args.confidence)
    
    # Print results
    if not detected:
        print("No subcircuits detected with the specified confidence threshold.")
    else:
        print(f"Detected {len(detected)} potential subcircuits:")
        for i, detection in enumerate(detected):
            print(f"  {i+1}. {detection['topology']} (confidence: {detection['confidence']:.2f})")
            print(f"     Size: {detection['size']} components")
            print(f"     Center component: {data.component_names[detection['center_node']] if hasattr(data, 'component_names') else detection['center_node']}")
            print(f"     Components: {[data.component_names[node] if hasattr(data, 'component_names') else node for node in detection['nodes'][:5]]}" + 
                  ("..." if len(detection['nodes']) > 5 else ""))
    
    # Visualize the results
    print(f"Generating visualization in {args.output}...")
    detector.visualize_detection(data, detected, save_path=args.output)
    print(f"Visualization saved to {args.output}")

if __name__ == "__main__":
    main() 
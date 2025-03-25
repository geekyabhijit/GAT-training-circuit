import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import random
from netlist_to_graph import load_netlist_dataset

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

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

# Define the GAT model
class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads1=8, heads2=1, dropout=0.6):
        super(GATModel, self).__init__()
        self.dropout = dropout
        
        # First GAT layer with 8 attention heads
        self.conv1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=heads1,
            dropout=dropout
        )
        
        # Second GAT layer with 1 attention head for classification
        self.conv2 = GATConv(
            in_channels=hidden_dim * heads1,
            out_channels=output_dim,
            heads=heads2,
            concat=False,
            dropout=dropout
        )

    def forward(self, x, edge_index):
        # Apply dropout to input features
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # First GAT layer with ELU activation
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # Apply dropout between layers
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GAT layer with softmax activation for classification
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

# Training function
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Evaluation function
def evaluate(model, loader, device):
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            predictions.extend(pred.cpu().numpy())
            labels.extend(data.y.cpu().numpy())
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    conf_matrix = confusion_matrix(labels, predictions)
    
    return accuracy, f1, recall, conf_matrix

def main():
    # Model hyperparameters
    input_dim = 4      # Node feature dimension (NMOS, PMOS, Resistor, Capacitor)
    hidden_dim = 8     # Hidden dimension per attention head
    output_dim = 8     # Number of circuit topology classes
    heads1 = 8         # Number of attention heads in first layer
    heads2 = 1         # Number of attention heads in second layer
    dropout = 0.6      # Dropout rate
    lr = 0.005         # Learning rate
    weight_decay = 0.0005  # L2 regularization
    epochs = 200       # Number of training epochs
    batch_size = 32    # Batch size
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Directory structure for netlist files
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
    
    # Create data directories if they don't exist
    for directory in data_dirs.values():
        os.makedirs(directory, exist_ok=True)
    
    # Load netlist data
    print("Loading netlist data...")
    dataset = load_netlist_dataset("data", data_dirs)
    
    if len(dataset) == 0:
        print("No netlist files found. Please add SPICE netlist files to the data directories.")
        print("Example data directory structure:")
        for topology, directory in data_dirs.items():
            print(f"  - {directory}/*.sp")
        return
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Split the dataset into training, validation, and test sets
    random.shuffle(dataset.data_list)  # Shuffle the dataset
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_data = dataset.data_list[:train_size]
    val_data = dataset.data_list[train_size:train_size+val_size]
    test_data = dataset.data_list[train_size+val_size:]
    
    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Initialize the model
    model = GATModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        heads1=heads1,
        heads2=heads2,
        dropout=dropout
    ).to(device)
    
    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Training loop
    best_val_acc = 0
    train_losses = []
    val_accs = []
    val_f1s = []
    val_recalls = []
    
    print("Starting training...")
    for epoch in tqdm(range(epochs)):
        # Train
        train_loss = train(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_acc, val_f1, val_recall, _ = evaluate(model, val_loader, device)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_recalls.append(val_recall)
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_circuit_gat_model.pth')
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:03d}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val Recall: {val_recall:.4f}')
    
    # Load the best model
    model.load_state_dict(torch.load('best_circuit_gat_model.pth'))
    
    # Test the model
    test_acc, test_f1, test_recall, conf_matrix = evaluate(model, test_loader, device)
    print(f'Test Results - Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}, Recall: {test_recall:.4f}')
    
    # Plot training progress
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(val_accs)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.subplot(2, 2, 3)
    plt.plot(val_f1s, label='F1 Score')
    plt.plot(val_recalls, label='Recall')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    # Plot confusion matrix
    plt.subplot(2, 2, 4)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[t[:10] for t in CIRCUIT_TOPOLOGIES], 
                yticklabels=[t[:10] for t in CIRCUIT_TOPOLOGIES])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('circuit_training_metrics.png')
    plt.show()
    
    # Print classification report for each topology
    for i, topology in enumerate(CIRCUIT_TOPOLOGIES):
        print(f"Class {i}: {topology}")
        true_positives = conf_matrix[i, i]
        false_positives = conf_matrix[:, i].sum() - true_positives
        false_negatives = conf_matrix[i, :].sum() - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main() 
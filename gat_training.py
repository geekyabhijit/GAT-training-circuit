import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)

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

# Function to generate synthetic analog circuit topology data
def generate_circuit_data(num_samples=200, num_classes=8):
    dataset = []
    
    # Define characteristics for each topology
    topology_specs = {
        0: {  # Simple Current Mirror
            "nodes": (3, 5),  # Number of nodes (min, max)
            "edges_density": 0.7,  # Connectivity density
            "feature_patterns": [[1, 0, 0, 0], [0, 1, 0, 0]]  # Common feature patterns
        },
        1: {  # Cascaded Current Mirror
            "nodes": (5, 8),
            "edges_density": 0.6,
            "feature_patterns": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        },
        2: {  # Differential Pair
            "nodes": (4, 7),
            "edges_density": 0.5,
            "feature_patterns": [[0, 1, 0, 0], [0, 0, 1, 0]]
        },
        3: {  # Single-Stage Differential Amplifier
            "nodes": (6, 10),
            "edges_density": 0.4,
            "feature_patterns": [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        },
        4: {  # Two-Stage Amplifier
            "nodes": (8, 15),
            "edges_density": 0.3,
            "feature_patterns": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        },
        5: {  # LDO
            "nodes": (7, 12),
            "edges_density": 0.35,
            "feature_patterns": [[1, 0, 0, 0], [0, 0, 0, 1]]
        },
        6: {  # Bandgap Reference
            "nodes": (8, 14),
            "edges_density": 0.45,
            "feature_patterns": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        },
        7: {  # Comparator
            "nodes": (5, 10),
            "edges_density": 0.4,
            "feature_patterns": [[0, 1, 0, 0], [0, 0, 1, 0]]
        }
    }
    
    # Generate examples for each topology
    samples_per_class = num_samples // num_classes
    for class_idx in range(num_classes):
        specs = topology_specs[class_idx]
        
        for _ in range(samples_per_class):
            # Determine number of nodes for this sample
            num_nodes = np.random.randint(specs["nodes"][0], specs["nodes"][1] + 1)
            
            # Generate node features - representing device types (NMOS, PMOS, Resistor, Capacitor)
            x = torch.zeros(num_nodes, 4)
            
            # Add some characteristic feature patterns for this topology
            for i in range(num_nodes):
                if i < len(specs["feature_patterns"]) and np.random.random() > 0.3:
                    # Use a characteristic pattern with high probability
                    x[i] = torch.tensor(specs["feature_patterns"][i % len(specs["feature_patterns"])])
                else:
                    # Random feature with low probability
                    x[i, np.random.randint(0, 4)] = 1
            
            # Create graph connectivity for this topology
            edge_index = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j and np.random.random() < specs["edges_density"]:
                        edge_index.append([i, j])
            
            if not edge_index:  # Ensure at least one edge
                i, j = 0, 1
                edge_index.append([i, j])
                edge_index.append([j, i])
                
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            
            # All nodes in this graph belong to the same topology class
            y = torch.tensor([class_idx] * num_nodes, dtype=torch.long)
            
            # Create a PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, y=y)
            dataset.append(data)
    
    return dataset

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

# Main function
def main():
    # Model hyperparameters
    input_dim = 4      # Node feature dimension (one-hot encoding for device types)
    hidden_dim = 8     # Hidden dimension per attention head
    output_dim = 8     # Number of circuit topology classes
    heads1 = 8         # Number of attention heads in first layer
    heads2 = 1         # Number of attention heads in second layer
    dropout = 0.6      # Dropout rate
    lr = 0.005         # Learning rate
    weight_decay = 0.0005  # L2 regularization
    epochs = 200       # Number of training epochs
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic circuit topology dataset
    print("Generating synthetic circuit topology data...")
    all_data = generate_circuit_data(num_samples=240, num_classes=output_dim)
    
    # Split the dataset into training, validation, and test sets
    train_data = all_data[:160]
    val_data = all_data[160:200]
    test_data = all_data[200:]
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
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
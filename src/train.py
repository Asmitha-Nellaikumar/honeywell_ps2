# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import your custom modules
from src.model import GraphAutoencoder
from src.build_graph import build_graph

def train_model(train_data, num_features, epochs=50, lr=0.001, model_save_path="outputs/models/graph_autoencoder.pt"):
    # Convert data to a PyTorch tensor
    x = torch.tensor(train_data, dtype=torch.float)

    # Build the graph on the entire training data
    # NOTE: The edge_index remains the same for the whole dataset.
    edge_index = build_graph(train_data)
    
    # --- KEY CHANGE: Use DataLoader for batching ---
    # Create a TensorDataset from your data
    dataset = TensorDataset(x)
    # Create a DataLoader to iterate over the data in batches
    # A batch size of 128 is a good starting point. You can adjust this.
    batch_size = 128
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GraphAutoencoder(num_features)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # --- KEY CHANGE: Iterate over batches from the DataLoader ---
        for i, batch_data in enumerate(loader):
            # batch_data is a list of tensors. We only need the first one (our features).
            x_batch = batch_data[0]
            
            optimizer.zero_grad()
            
            # Pass the batch through the model. The edge_index is the same for all batches.
            output, _ = model(x_batch, edge_index)
            
            loss = criterion(output, x_batch)  # reconstruction loss on the batch
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_save_path)
    return model, edge_index

# (The rest of your code from main.py, infer.py, etc., remains the same)

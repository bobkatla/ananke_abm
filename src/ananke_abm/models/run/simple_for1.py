import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torchdiffeq import odeint
import torch_geometric

class SimpleZoneODE(nn.Module):
    """Simple neural ODE for zone dynamics"""
    
    def __init__(self, zone_features_dim=7, person_attrs_dim=8, hidden_dim=32, num_zones=8):
        super().__init__()
        
        self.num_zones = num_zones
        self.hidden_dim = hidden_dim
        
        # Graph neural network for zone embeddings - fix the Sequential usage
        self.gnn_conv1 = pyg_nn.GCNConv(zone_features_dim, hidden_dim)
        self.gnn_conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        
        # Time encoding
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Dynamics network
        self.dynamics_net = nn.Sequential(
            nn.Linear(hidden_dim + person_attrs_dim + 16, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # Output: velocity in embedding space
        )
        
        # Store graph data for ODE computation
        self.zone_features = None
        self.edge_index = None
        self.person_attrs = None
        
    def set_graph_data(self, zone_features, edge_index, person_attrs):
        """Set graph data that stays constant during ODE solving"""
        self.zone_features = zone_features
        self.edge_index = edge_index  
        self.person_attrs = person_attrs
        
    def forward(self, t, zone_embedding):
        """Compute dz/dt - simplified signature for odeint"""
        
        # Update zone embeddings with graph structure
        x = self.gnn_conv1(self.zone_features, self.edge_index)
        x = self.activation(x)
        updated_zone_embeddings = self.gnn_conv2(x, self.edge_index)
        updated_zone_embeddings = self.activation(updated_zone_embeddings)
        
        # Time encoding - handle both scalar and tensor time
        if t.dim() == 0:  # scalar
            t_input = t.unsqueeze(0).unsqueeze(0)
        else:
            t_input = t.view(-1, 1)
        
        time_vec = self.time_encoder(t_input)
        if zone_embedding.dim() == 1:
            time_vec = time_vec.squeeze(0)
        else:
            time_vec = time_vec.expand(zone_embedding.shape[0], -1)
        
        # Expand person attributes to match zone_embedding dimensions
        if zone_embedding.dim() == 1:
            person_expanded = self.person_attrs
        else:
            person_expanded = self.person_attrs.unsqueeze(0).expand(zone_embedding.shape[0], -1)
        
        # Combine inputs
        combined_input = torch.cat([
            zone_embedding,  # Current position in zone embedding space
            person_expanded,  # Person characteristics
            time_vec  # Time context
        ], dim=-1)
        
        # Compute dynamics
        velocity = self.dynamics_net(combined_input)
        
        return velocity

class ZoneTrajectoryPredictor(nn.Module):
    """Complete trajectory prediction system"""
    
    def __init__(self, ode_func, num_zones=8, hidden_dim=32):
        super().__init__()
        
        self.ode_func = ode_func
        self.num_zones = num_zones
        self.hidden_dim = hidden_dim
        
        # Learn initial state from person attributes
        self.initial_state_net = nn.Sequential(
            nn.Linear(8, 32),  # person_attrs_dim
            nn.ReLU(),
            nn.Linear(32, hidden_dim)
        )
        
        # Decode zone embedding to zone probabilities
        self.zone_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_zones),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, person_attrs, times, zone_features, edge_index):
        """Predict full trajectory"""
        
        # Set graph data in ODE function
        self.ode_func.set_graph_data(zone_features, edge_index, person_attrs)
        
        # Initial state from person attributes  
        initial_embedding = self.initial_state_net(person_attrs)
        
        # Solve ODE for trajectory
        trajectory_embeddings = odeint(
            self.ode_func,
            initial_embedding,
            times,
            method='rk4',
            rtol=1e-4
        )
        
        # Decode to zone probabilities
        zone_probs = self.zone_decoder(trajectory_embeddings)
        
        return zone_probs, trajectory_embeddings

def train_zone_model(model, training_data, epochs=1000):
    """Train the zone trajectory model"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training data
    person_attrs = training_data["person_attrs"]
    times = training_data["times"]
    zone_targets = training_data["zone_observations"]  # Already 0-indexed
    zone_features = training_data["zone_features"]
    edge_index = training_data["edge_index"]
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        try:
            # Forward pass
            zone_probs, trajectory_embeddings = model(
                person_attrs, times, zone_features, edge_index
            )
            
            # Loss: match observed zones (already 0-indexed)
            loss = loss_fn(zone_probs, zone_targets)
            
            # Add smoothness regularization
            if len(trajectory_embeddings) > 1:
                velocity = trajectory_embeddings[1:] - trajectory_embeddings[:-1]
                acceleration = velocity[1:] - velocity[:-1] 
                smoothness_loss = torch.mean(acceleration ** 2)
                loss += 0.01 * smoothness_loss  # Reduce weight to avoid overly smooth
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
                
                # Show predictions
                predicted_zones = torch.argmax(zone_probs, dim=1)  # Keep 0-indexed
                accuracy = torch.mean((predicted_zones == zone_targets).float()).item()
                print(f"Accuracy: {accuracy:.3f}")
                print(f"Target zones:    {zone_targets[:10].tolist()}...")
                print(f"Predicted zones: {predicted_zones[:10].tolist()}...")
                print()
                
        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    return losses

def predict_and_visualize(model, training_data, sarah_schedule):
    """Generate predictions and visualize results"""
    
    model.eval()
    
    with torch.no_grad():
        # Make prediction
        zone_probs, trajectory_embeddings = model(
            training_data["person_attrs"],
            training_data["times"], 
            training_data["zone_features"],
            training_data["edge_index"]
        )
        
        predicted_zones = torch.argmax(zone_probs, dim=1)  # Keep 0-indexed for comparison
        actual_zones = training_data["zone_observations"]  # Already 0-indexed
        times = training_data["times"]
        
    # Print results
    print("=== PREDICTION RESULTS ===")
    print(f"{'Time':<6} {'Actual':<8} {'Predicted':<10} {'Probability':<12} {'Activity'}")
    print("-" * 70)
    
    for i, (time, actual, predicted, prob_dist) in enumerate(zip(
        times, actual_zones, predicted_zones, zone_probs
    )):
        max_prob = torch.max(prob_dist).item()
        activity = sarah_schedule[i]["activity"] if i < len(sarah_schedule) else "unknown"
        
        # Convert back to 1-indexed for display
        actual_display = actual.item() + 1
        predicted_display = predicted.item() + 1
        
        print(f"{time.item():5.1f}h {actual_display:<8} {predicted_display:<10} {max_prob:7.3f}     {activity}")
    
    # Calculate accuracy
    accuracy = torch.mean((predicted_zones == actual_zones).float()).item()
    print(f"\nAccuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    return predicted_zones, zone_probs

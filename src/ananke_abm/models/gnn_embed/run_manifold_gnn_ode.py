#!/usr/bin/env python3
"""
Runner for Manifold-Based GNN-ODE - Continuous Movement on Graph Manifold
Test the new formulation that models real continuous movement with hard physics constraints.
"""

import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

# Add to path to import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_generator.load_data import load_mobility_data
from models.gnn_embed.HomoGraph import HomoGraph
from models.gnn_embed.manifold_gnn_ode import (
    ManifoldGNNPhysicsODE, 
    ManifoldGNNODETrainer,
    GraphManifoldState
)

def create_manifold_graphs():
    """Create homogeneous graphs for manifold model"""
    print("📊 Creating manifold graphs...")
    
    # Load data
    trajectories, people_data, zones_data = load_mobility_data()
    
    # Create location graph DataFrame format
    zone_node_features = zones_data.set_index('zone_id')[['zone_type_retail', 'zone_type_residential', 
                                                         'zone_type_office', 'zone_type_recreation',
                                                         'zone_type_transport', 'x_coord', 'y_coord']]
    zone_node_features.index.name = 'node_id'
    
    # Create edge features for physical adjacency
    adjacency_edges = [
        (1, 2), (1, 3), (1, 4),  # Zone 1 connects to 2, 3, 4
        (2, 1), (2, 5), (2, 6),  # Zone 2 connects to 1, 5, 6  
        (3, 1), (3, 4), (3, 7),  # Zone 3 connects to 1, 4, 7
        (4, 1), (4, 3), (4, 8),  # Zone 4 connects to 1, 3, 8
        (5, 2), (5, 6), (5, 7),  # Zone 5 connects to 2, 6, 7
        (6, 2), (6, 5), (6, 8),  # Zone 6 connects to 2, 5, 8
        (7, 3), (7, 5), (7, 8),  # Zone 7 connects to 3, 5, 8
        (8, 4), (8, 6), (8, 7),  # Zone 8 connects to 4, 6, 7
    ]
    
    zone_edge_data = []
    for u, v in adjacency_edges:
        zone_edge_data.append({
            'edge_id': f"{u}_{v}",
            'distance': float(1.0),  # Uniform distance for simplicity
            'travel_time': float(1.0)
        })
    
    zone_edge_features = pd.DataFrame(zone_edge_data).set_index('edge_id')
    # Ensure all columns are numeric
    zone_edge_features = zone_edge_features.astype(float)
    
    # Create location graph
    location_graph = HomoGraph(zone_node_features, zone_edge_features, bidirectional=False)
    
    # Create person graph DataFrame format  
    person_node_features = people_data.set_index('person_id')[['age', 'income', 'home_zone_id', 'work_zone_id']]
    person_node_features.index.name = 'node_id'
    
    # Person graph has no edges (independent people)
    # Create a DataFrame with explicit numeric dtype to avoid object dtype issues
    person_edge_features = pd.DataFrame({
        'dummy_feature': pd.Series(dtype='float64')
    })
    person_edge_features.index.name = 'edge_id'
    
    # Create person graph
    person_graph = HomoGraph(person_node_features, person_edge_features, bidirectional=False)
    
    print(f"📍 Location graph: {location_graph.num_nodes} zones, {location_graph.num_edges} edges")
    print(f"👥 Person graph: {person_graph.num_nodes} people")
    
    return location_graph, person_graph, trajectories

def prepare_manifold_trajectories(trajectories) -> Dict:
    """Prepare trajectory data for manifold training"""
    
    print("🎯 Preparing manifold trajectories...")
    
    manifold_trajectories = {}
    
    for person_name, traj_data in trajectories.items():
        person_id = traj_data["person_id"]
        times = torch.tensor(traj_data["times"], dtype=torch.float32)
        zones = torch.tensor(traj_data["zones"], dtype=torch.long)
        
        # Convert 1-indexed zones to 0-indexed for model
        zones = zones - 1
        
        manifold_trajectories[person_name] = {
            "person_id": person_id,
            "times": times,
            "zones": zones
        }
    
    print(f"✅ Prepared {len(manifold_trajectories)} manifold trajectories")
    return manifold_trajectories

def demonstrate_manifold_concepts(model: ManifoldGNNPhysicsODE):
    """Demonstrate key manifold concepts"""
    
    print("\n🧮 Demonstrating Manifold Concepts:")
    print("=" * 50)
    
    # 1. Pure states
    print("1. Pure States (single zone occupancy):")
    pure_state_zone1 = GraphManifoldState.from_zone_idx(0, 8)  # Zone 1 (0-indexed)
    pure_embedding = pure_state_zone1.to_embedding(model.zone_embeddings)
    print(f"   Zone 1 pure state weights: {pure_state_zone1.weights}")
    print(f"   Dominant zone: {pure_state_zone1.get_dominant_zone() + 1}")  # Convert back to 1-indexed
    
    # 2. Transition states
    print("\n2. Transition States (between zones):")
    transition_state = GraphManifoldState.from_transition(0, 1, 0.3, 8)  # 30% progress from zone 1 to 2
    transition_embedding = transition_state.to_embedding(model.zone_embeddings)
    print(f"   Transition state weights: {transition_state.weights}")
    print(f"   Active zones: {transition_state.get_active_zones() + 1}")  # Convert to 1-indexed
    
    # 3. Physics constraints
    print("\n3. Physics Constraints (adjacency matrix):")
    print("   Adjacency Matrix (1=connected, 0=disconnected):")
    adj_matrix = model.adjacency_matrix.numpy()
    print("     ", " ".join([f"Z{i+1}" for i in range(8)]))
    for i in range(8):
        print(f"   Z{i+1}", " ".join([f"{int(adj_matrix[i,j]):2d}" for j in range(8)]))
    
    # 4. Allowed flows
    print("\n4. Allowed Flow Examples:")
    test_weights = torch.zeros(8)
    test_weights[0] = 0.7  # 70% in zone 1
    test_weights[1] = 0.3  # 30% in zone 2
    allowed = model.get_allowed_flows(test_weights)
    print(f"   From state (70% Zone1, 30% Zone2):")
    print(f"   Allowed flows to: {torch.where(allowed > 0)[0] + 1}")  # Convert to 1-indexed
    
    print("=" * 50)

def train_manifold_model():
    """Train the manifold-based GNN-ODE model"""
    
    print("🚀 Training Manifold-Based GNN-ODE")
    print("=" * 60)
    
    # Create graphs and load data
    location_graph, person_graph, raw_trajectories = create_manifold_graphs()
    trajectories = prepare_manifold_trajectories(raw_trajectories)
    
    # Create model
    model = ManifoldGNNPhysicsODE(
        location_graph=location_graph,
        person_graph=person_graph,
        embedding_dim=64,
        num_zones=8
    )
    
    print(f"\n📊 Model Architecture:")
    print(f"   Zone embeddings: {model.zone_embeddings.shape}")
    print(f"   Person embedder: {sum(p.numel() for p in model.person_embedder.parameters())} params")
    print(f"   Flow network: {sum(p.numel() for p in model.flow_net.parameters())} params")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Demonstrate concepts
    demonstrate_manifold_concepts(model)
    
    # Create trainer
    trainer = ManifoldGNNODETrainer(model, lr=0.001, save_dir="saved_models/manifold")
    
    # Train model
    print(f"\n🎯 Training on {len(trajectories)} trajectories...")
    trainer.train(trajectories, num_epochs=1000, verbose=True)
    
    return model, trainer, trajectories, location_graph, person_graph

def evaluate_manifold_model(model, trainer, trajectories):
    """Evaluate manifold model performance"""
    
    print("\n📈 Evaluating Manifold Model:")
    print("=" * 50)
    
    # Load best model
    loaded = trainer.load_best_model(zero_violations_priority=True)
    if not loaded:
        print("❌ No saved model found, using current model")
    
    # Evaluate
    results = trainer.evaluate(trajectories)
    
    # Analyze results
    total_accuracy = []
    total_violations = 0
    
    print("\n👤 Individual Results:")
    for person_name, result in results.items():
        accuracy = result['accuracy']
        violations = model._count_physics_violations(result['predicted_zones'])
        
        total_accuracy.append(accuracy)
        total_violations += violations
        
        print(f"   {person_name}: {accuracy:.1%} accuracy, {violations} violations")
    
    # Overall statistics
    mean_accuracy = np.mean(total_accuracy)
    violation_rate = total_violations / sum(len(r['predicted_zones']) for r in results.values())
    
    print(f"\n🎯 Overall Performance:")
    print(f"   Mean Accuracy: {mean_accuracy:.1%}")
    print(f"   Physics Violations: {total_violations} ({violation_rate:.1%} rate)")
    print(f"   Zero-Violation Success: {'✅ YES' if total_violations == 0 else '❌ NO'}")
    
    return results

def visualize_manifold_results(trainer, results):
    """Create comprehensive visualizations"""
    
    print("\n📊 Creating Visualizations...")
    
    # Load training data
    save_dir = Path("saved_models/manifold")
    
    try:
        training_losses = np.load(save_dir / "manifold_gnn_ode_training_losses.npy")
        physics_violations = np.load(save_dir / "manifold_gnn_ode_physics_violations.npy")
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Manifold GNN-ODE Results', fontsize=16, fontweight='bold')
        
        # Training loss
        axes[0,0].plot(training_losses, 'b-', linewidth=2)
        axes[0,0].set_title('Training Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_yscale('log')
        
        # Physics violations
        axes[0,1].plot(physics_violations, 'r-', linewidth=2)
        axes[0,1].set_title('Physics Violations Over Training')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Number of Violations')
        axes[0,1].grid(True, alpha=0.3)
        
        # Individual accuracies
        people = list(results.keys())
        accuracies = [results[p]['accuracy'] for p in people]
        
        bars = axes[1,0].bar(people, accuracies, color=['skyblue', 'lightcoral'])
        axes[1,0].set_title('Individual Trajectory Accuracy')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].set_ylim(0, 1)
        plt.setp(axes[1,0].get_xticklabels(), rotation=45)
        
        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{acc:.1%}', ha='center', va='bottom')
        
        # Example trajectory comparison
        person_name = list(results.keys())[0]  # First person
        result = results[person_name]
        
        times = result['times'].numpy()
        observed = result['observed_zones'].numpy() + 1  # Convert to 1-indexed
        predicted = result['predicted_zones'].numpy() + 1
        
        axes[1,1].plot(times, observed, 'o-', label='Observed', linewidth=2, markersize=6)
        axes[1,1].plot(times, predicted, 's--', label='Predicted', linewidth=2, markersize=6)
        axes[1,1].set_title(f'Example Trajectory: {person_name}')
        axes[1,1].set_xlabel('Time (hours)')
        axes[1,1].set_ylabel('Zone')
        axes[1,1].set_ylim(0.5, 8.5)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path("manifold_gnn_ode_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Results saved to: {plot_path}")
        
        plt.show()
        
    except FileNotFoundError as e:
        print(f"❌ Could not load training data: {e}")

def main():
    """Main function to run manifold GNN-ODE"""
    
    print("🌟 Manifold-Based GNN-ODE: Continuous Movement on Graph Manifold")
    print("=" * 80)
    print("This approach models REAL continuous movement with:")
    print("  🔵 States as probability distributions over zones (convex combinations)")
    print("  🔵 Hard physics constraints (only flow along graph edges)")  
    print("  🔵 True continuous trajectories during zone transitions")
    print("  🔵 Zone embeddings as basis vectors for manifold")
    print("=" * 80)
    
    try:
        # Train model
        model, trainer, trajectories, location_graph, person_graph = train_manifold_model()
        
        # Evaluate model
        results = evaluate_manifold_model(model, trainer, trajectories)
        
        # Visualize results
        visualize_manifold_results(trainer, results)
        
        print("\n🎉 Manifold GNN-ODE Run Complete!")
        print("Key benefits of this approach:")
        print("  ✅ Physically meaningful continuous states")
        print("  ✅ Hard physics constraints (impossible flows blocked)")
        print("  ✅ Rich state space (convex hull of zone embeddings)")
        print("  ✅ Realistic transition dynamics")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
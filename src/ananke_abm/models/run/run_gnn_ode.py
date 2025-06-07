from ananke_abm.data_generator.mock_2p import create_two_person_training_data
from ananke_abm.data_generator.mock_locations import create_mock_zone_graph
from ananke_abm.models.gnn_embed.HomoGraph import HomoGraph
from ananke_abm.models.gnn_embed.gnn_ode import GNNPhysicsODE, GNNODETrainer
import pandas as pd
import torch
import numpy as np

def create_location_graph():
    """Create HomoGraph for locations with static spatial information only"""
    
    # Get location data
    zone_graph, zone_data = create_mock_zone_graph()
    
    # Create node features DataFrame (static location attributes)
    node_data = []
    for zone_id in sorted(zone_graph.nodes()):
        zone_info = zone_graph.nodes[zone_id]
        node_data.append({
            'population_norm': zone_info["population"] / 10000.0,
            'job_opportunities_norm': zone_info["job_opportunities"] / 5000.0,
            'retail_accessibility': zone_info["retail_accessibility"],
            'transit_accessibility': zone_info["transit_accessibility"],
            'attractiveness': zone_info["attractiveness"],
            'coord_x_norm': zone_info["coordinates"][0] / 5.0,
            'coord_y_norm': zone_info["coordinates"][1] / 5.0,
        })
    
    node_features_df = pd.DataFrame(
        node_data, 
        index=pd.Index(sorted(zone_graph.nodes()), name='node_id')
    )
    
    # Create edge features DataFrame (static connectivity attributes)
    edge_data = []
    edge_ids = []
    for u, v, edge_info in zone_graph.edges(data=True):
        edge_ids.append((u, v))
        edge_data.append({
            'distance_norm': edge_info["distance"] / 10.0,
            'travel_time_norm': edge_info["travel_time"] / 60.0,
        })
    
    edge_features_df = pd.DataFrame(
        edge_data,
        index=pd.Index([f"{u}_{v}" for u, v in edge_ids], name='edge_id')
    )
    
    return HomoGraph(node_features_df, edge_features_df)

def create_person_graph():
    """Create HomoGraph for people with static demographic information only"""
    
    # Get person data
    sarah_data, marcus_data = create_two_person_training_data()
    
    # Extract static person attributes only (no trajectory data)
    sarah_features = sarah_data["person_attrs"].numpy()
    marcus_features = marcus_data["person_attrs"].numpy()
    
    feature_names = [
        'age_norm', 'income_norm', 'employment_full_time', 'commute_car',
        'activity_flexibility', 'social_tendency', 'household_size_norm', 'has_car'
    ]
    
    node_features_df = pd.DataFrame(
        [sarah_features, marcus_features],
        columns=feature_names,
        index=pd.Index([sarah_data["person_id"], marcus_data["person_id"]], name='node_id')
    )
    
    # Create edges based on static relationships (could be social network, etc.)
    # For now, simple demographic similarity
    age_sim = 1 - abs(sarah_features[0] - marcus_features[0])
    income_sim = 1 - abs(sarah_features[1] - marcus_features[1])
    overall_sim = (age_sim + income_sim) / 2
    
    edge_features_df = pd.DataFrame(
        [[overall_sim, 0.3]],  # similarity, interaction_strength
        columns=['demographic_similarity', 'social_connection'],
        index=pd.Index(["1_2"], name='edge_id')
    )
    
    return HomoGraph(node_features_df, edge_features_df)

def extract_trajectory_data():
    """Extract trajectory data separately - this is what the ODE model will fit"""
    
    sarah_data, marcus_data = create_two_person_training_data()
    
    # Return trajectory snapshots (the temporal data to be modeled)
    trajectories = {
        "sarah": {
            "person_id": sarah_data["person_id"],
            "times": sarah_data["times"],
            "zones": sarah_data["zone_observations"],
            "name": sarah_data["person_name"]
        },
        "marcus": {
            "person_id": marcus_data["person_id"],
            "times": marcus_data["times"],
            "zones": marcus_data["zone_observations"],
            "name": marcus_data["person_name"]
        }
    }
    
    return trajectories

def print_trajectory_comparison(results: dict):
    """Print detailed comparison of predicted vs observed trajectories"""
    
    for person_name, result in results.items():
        print(f"\n=== {person_name.upper()} TRAJECTORY ===")
        print(f"Accuracy: {result['accuracy']:.3f}")
        
        observed = result['observed_zones']
        predicted = result['predicted_zones']
        times = result['times']
        
        print(f"{'Time':>6} {'Observed':>8} {'Predicted':>9} {'Match':>5}")
        print("-" * 35)
        
        for i in range(min(15, len(times))):  # Show first 15 points
            time_str = f"{times[i].item():.2f}"
            obs_zone = observed[i].item()
            pred_zone = predicted[i].item()
            match = "✓" if obs_zone == pred_zone else "✗"
            
            print(f"{time_str:>6} {obs_zone:>8} {pred_zone:>9} {match:>5}")
        
        if len(times) > 15:
            print(f"... ({len(times) - 15} more time points)")

def check_physics_violations(results: dict, location_graph: HomoGraph):
    """Check if any predicted transitions violate graph connectivity"""
    
    edge_index = location_graph.edge_index
    
    # Create adjacency set for fast lookup
    adjacency = set()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        adjacency.add((u, v))
        adjacency.add((v, u))  # Bidirectional
    
    # Add self-loops (can stay in same zone)
    for i in range(8):  # 8 zones
        adjacency.add((i, i))
    
    print("\n=== PHYSICS VIOLATION CHECK ===")
    
    total_transitions = 0
    total_violations = 0
    
    for person_name, result in results.items():
        predicted = result['predicted_zones']
        violations = 0
        transitions = 0
        
        print(f"\n{person_name.upper()} Physics Check:")
        
        for i in range(len(predicted) - 1):
            current_zone = predicted[i].item()
            next_zone = predicted[i + 1].item()
            transitions += 1
            
            # Check if transition is allowed
            if (current_zone, next_zone) not in adjacency:
                violations += 1
                times = result['times']
                print(f"  ❌ VIOLATION: Zone {current_zone} → {next_zone} at t={times[i+1].item():.2f}")
        
        violation_rate = violations / transitions if transitions > 0 else 0
        print(f"  Total transitions: {transitions}")
        print(f"  Physics violations: {violations} ({violation_rate:.1%})")
        
        total_transitions += transitions
        total_violations += violations
    
    overall_violation_rate = total_violations / total_transitions if total_transitions > 0 else 0
    print(f"\n=== OVERALL PHYSICS SUMMARY ===")
    print(f"Total transitions: {total_transitions}")
    print(f"Physics violations: {total_violations} ({overall_violation_rate:.1%})")
    
    if overall_violation_rate == 0:
        print("🎉 Perfect! No physics violations detected.")
    elif overall_violation_rate < 0.1:
        print("✅ Good physics compliance (< 10% violations)")
    elif overall_violation_rate < 0.3:
        print("⚠️  Moderate physics violations (10-30%)")
    else:
        print("❌ High physics violations (> 30%) - constraints not working well")
    
    return total_violations, total_transitions

def show_graph_connectivity(location_graph: HomoGraph):
    """Show the allowed transitions for reference"""
    
    edge_index = location_graph.edge_index
    
    print("\n=== GRAPH CONNECTIVITY REFERENCE ===")
    
    # Build adjacency lists
    adjacency_lists = {i: set() for i in range(8)}
    
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        adjacency_lists[u].add(v)
        adjacency_lists[v].add(u)
    
    # Add self-loops
    for i in range(8):
        adjacency_lists[i].add(i)
    
    for zone in range(8):
        neighbors = sorted(list(adjacency_lists[zone]))
        print(f"Zone {zone}: can transition to {neighbors}")
    
    total_allowed_transitions = sum(len(neighbors) for neighbors in adjacency_lists.values())
    print(f"\nTotal allowed transitions: {total_allowed_transitions} out of {8*8} possible")

def train_and_evaluate_model():
    """Complete training and evaluation pipeline"""
    
    print("=== GNN-ODE Training & Evaluation ===")
    
    # Create graphs and trajectory data
    print("\n1. Loading data...")
    location_graph = create_location_graph()
    person_graph = create_person_graph()
    trajectories = extract_trajectory_data()
    
    print(f"   Location graph: {location_graph}")
    print(f"   Person graph: {person_graph}")
    print(f"   Trajectories: {list(trajectories.keys())}")
    
    # Create model
    print("\n2. Creating GNN-ODE model...")
    model = GNNPhysicsODE(
        location_graph=location_graph,
        person_graph=person_graph,
        embedding_dim=64,
        num_gnn_layers=2
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    print(f"   Location embeddings shape: {model.location_embeddings.shape}")
    print(f"   Person embeddings shape: {model.person_embeddings.shape}")
    
    # Create trainer
    trainer = GNNODETrainer(model, lr=0.001)
    
    # Train model
    print("\n3. Training model...")
    trainer.train(trajectories, num_epochs=100, verbose=True)
    
    # Evaluate model
    print("\n4. Evaluating model...")
    results = trainer.evaluate(trajectories)
    
    # Print results
    print("\n5. Results Summary:")
    for person_name, result in results.items():
        print(f"   {person_name}: {result['accuracy']:.1%} accuracy")
    
    # Detailed comparison
    print_trajectory_comparison(results)
    
    return model, trainer, results

if __name__ == "__main__":
    # Test the clean separation first
    print("=== Clean Graph + Trajectory Separation ===")

    # Create static graphs
    print("\n=== Creating Location Graph (Static) ===")
    location_graph = create_location_graph()
    print(f"Location graph: {location_graph}")
    print(f"Location schema: {location_graph.extract_schema()}")

    print("\n=== Creating Person Graph (Static) ===")
    person_graph = create_person_graph()
    print(f"Person graph: {person_graph}")
    print(f"Person schema: {person_graph.extract_schema()}")

    print("\n=== Extracting Trajectory Data (Dynamic) ===")
    trajectories = extract_trajectory_data()
    print(f"Sarah trajectory: {len(trajectories['sarah']['times'])} time points")
    print(f"Marcus trajectory: {len(trajectories['marcus']['times'])} time points")

    # Show first few trajectory points
    print(f"\nSarah first 5 points: {list(zip(trajectories['sarah']['times'][:5], trajectories['sarah']['zones'][:5]))}")
    print(f"Marcus first 5 points: {list(zip(trajectories['marcus']['times'][:5], trajectories['marcus']['zones'][:5]))}")

    print("\n=== Architecture Summary ===")
    print("✓ Location Graph: Static spatial features + connectivity")
    print("✓ Person Graph: Static demographic features + relationships") 
    print("✓ Trajectory Data: Time-series of location visits (what ODE models)")
    print("✓ Clean separation: Structure vs Dynamics")
    print("✓ Ready for GNN-ODE: Graphs provide context, trajectories provide training data")
    
    print("\n" + "="*60)
    
    # Train and evaluate the model
    model, trainer, results = train_and_evaluate_model()
    
    print("\n=== Final Summary ===")
    print("✓ GNN embeddings capture location and person features")
    print("✓ ODE learns movement dynamics in embedding space")
    print("✓ Physics constraints from graph connectivity")
    print("✓ Model trained and evaluated on 2-person dataset")
    
    overall_accuracy = np.mean([r['accuracy'] for r in results.values()])
    print(f"✓ Overall accuracy: {overall_accuracy:.1%}")
    
    if overall_accuracy > 0.8:
        print("🎉 Great performance! Model learned the movement patterns well.")
    elif overall_accuracy > 0.6:
        print("👍 Good performance! Model captured major movement patterns.")
    else:
        print("📊 Model is learning - may need more training or tuning.")

    # Check physics violations
    total_violations, total_transitions = check_physics_violations(results, location_graph)

    # Show graph connectivity for reference
    show_graph_connectivity(location_graph)
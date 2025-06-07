from ananke_abm.data_generator.mock_2p import create_two_person_training_data
from ananke_abm.data_generator.mock_locations import create_mock_zone_graph
from ananke_abm.models.gnn_embed.HomoGraph import HomoGraph
import pandas as pd
import torch

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

# Test the clean separation
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
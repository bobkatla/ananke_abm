#!/usr/bin/env python3
"""
Data loading functions for GNN-ODE models
Converts mock data to expected format for training
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .mock_2p import create_two_person_training_data
from .mock_locations import create_mock_zone_graph

def load_mobility_data() -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Load mobility data in the format expected by GNN-ODE models
    
    Returns:
        trajectories: Dict with person trajectory data
        people_data: DataFrame with person information
        zones_data: DataFrame with zone information
    """
    
    # Get mock data
    sarah_data, marcus_data = create_two_person_training_data()
    zone_graph, zones_raw, _ = create_mock_zone_graph()
    
    # Format trajectories
    trajectories = {}
    
    # Add Sarah's trajectory
    trajectories["Sarah"] = {
        "person_id": sarah_data["person_id"],
        "times": sarah_data["times"].numpy(),
        "zones": sarah_data["zone_observations"].numpy() + 1  # Convert back to 1-indexed
    }
    
    # Add Marcus's trajectory  
    trajectories["Marcus"] = {
        "person_id": marcus_data["person_id"],
        "times": marcus_data["times"].numpy(),
        "zones": marcus_data["zone_observations"].numpy() + 1  # Convert back to 1-indexed
    }
    
    # Create people DataFrame
    people_data = pd.DataFrame([
        {
            "person_id": int(sarah_data["person_id"]),
            "name": "Sarah",
            "age": float(sarah_data["person_attrs"][0].item() * 100),  # Denormalize
            "income": float(sarah_data["person_attrs"][1].item() * 100000),  # Denormalize
            "home_zone_id": int(1),  # Sarah's home
            "work_zone_id": int(5)   # Sarah's work
        },
        {
            "person_id": int(marcus_data["person_id"]),
            "name": "Marcus", 
            "age": float(marcus_data["person_attrs"][0].item() * 100),  # Denormalize
            "income": float(marcus_data["person_attrs"][1].item() * 100000),  # Denormalize
            "home_zone_id": int(3),  # Marcus's home
            "work_zone_id": int(6)   # Marcus's work
        }
    ])
    
    # Create zones DataFrame
    zones_data_list = []
    for zone_id, zone_info in zones_raw.items():
        # Determine zone type flags
        zone_type = zone_info["type"]
        zones_data_list.append({
            "zone_id": int(zone_id),
            "name": zone_info["name"],
            "zone_type_retail": int(1 if "retail" in zone_type else 0),
            "zone_type_residential": int(1 if "residential" in zone_type else 0),
            "zone_type_office": int(1 if "office" in zone_type or "commercial" in zone_type else 0),
            "zone_type_recreation": int(1 if zone_type in ["recreation", "park", "entertainment"] else 0),
            "zone_type_transport": int(0),  # None in our mock data
            "x_coord": float(zone_info["coordinates"][0]),
            "y_coord": float(zone_info["coordinates"][1]),
            "population": float(zone_info["population"]),
            "job_opportunities": float(zone_info["job_opportunities"]),
            "retail_accessibility": float(zone_info["retail_accessibility"]),
            "transit_accessibility": float(zone_info["transit_accessibility"]),
            "attractiveness": float(zone_info["attractiveness"])
        })
    
    zones_data = pd.DataFrame(zones_data_list)
    
    print(f"📊 Loaded mobility data:")
    print(f"   🚶 {len(trajectories)} people with trajectories")
    print(f"   📍 {len(zones_data)} zones")
    print(f"   ⏱️  {sum(len(t['times']) for t in trajectories.values())} total time points")
    
    return trajectories, people_data, zones_data

def get_zone_adjacency_matrix() -> np.ndarray:
    """Get the zone adjacency matrix for physics constraints"""
    zone_graph, _, _ = create_mock_zone_graph()
    
    # Create adjacency matrix
    num_zones = len(zone_graph.nodes())
    adjacency = np.zeros((num_zones, num_zones))
    
    # Add edges (bidirectional)
    for u, v in zone_graph.edges():
        adjacency[u-1, v-1] = 1  # Convert to 0-indexed
        adjacency[v-1, u-1] = 1
    
    # Add self-loops (can stay in same zone)
    for i in range(num_zones):
        adjacency[i, i] = 1
    
    return adjacency

def print_data_summary():
    """Print a summary of the loaded data"""
    trajectories, people_data, zones_data = load_mobility_data()
    
    print("\n📋 Data Summary:")
    print("=" * 50)
    
    print("\n👥 People:")
    for _, person in people_data.iterrows():
        print(f"   {person['name']} (ID: {person['person_id']})")
        print(f"      Age: {person['age']:.0f}, Income: ${person['income']:,.0f}")
        print(f"      Home: Zone {person['home_zone_id']}, Work: Zone {person['work_zone_id']}")
    
    print("\n📍 Zones:")
    for _, zone in zones_data.iterrows():
        zone_types = []
        if zone['zone_type_retail']: zone_types.append('retail')
        if zone['zone_type_residential']: zone_types.append('residential')
        if zone['zone_type_office']: zone_types.append('office')
        if zone['zone_type_recreation']: zone_types.append('recreation')
        
        print(f"   Zone {zone['zone_id']}: {zone['name']}")
        print(f"      Types: {', '.join(zone_types) if zone_types else 'other'}")
        print(f"      Population: {zone['population']:,}")
    
    print("\n🚶 Trajectories:")
    for person_name, traj in trajectories.items():
        times = traj['times']
        zones = traj['zones']
        print(f"   {person_name}: {len(times)} time points")
        print(f"      Time range: {times[0]:.1f} - {times[-1]:.1f} hours")
        print(f"      Zones visited: {sorted(set(zones))}")
        print(f"      Zone transitions: {len(set(zip(zones[:-1], zones[1:])))}")
    
    print("=" * 50)

if __name__ == "__main__":
    print_data_summary() 
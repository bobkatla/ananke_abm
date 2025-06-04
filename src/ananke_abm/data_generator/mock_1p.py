# create mock_data.py
import torch
import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class Person:
    """Mock person with realistic attributes"""
    person_id: int = 1
    name: str = "Sarah Chen"
    
    # Demographics
    age: int = 32
    income: float = 75000  # Annual income in USD
    employment_status: str = "full_time"
    occupation: str = "software_engineer"
    
    # Behavioral attributes
    commute_preference: str = "car"  # car, public_transit, bike, walk
    activity_flexibility: float = 0.3  # 0=rigid schedule, 1=very flexible
    social_tendency: float = 0.6  # 0=homebody, 1=very social
    
    # Household attributes (even though single-person household)
    household_income: float = 75000
    household_size: int = 1
    dwelling_type: str = "apartment"
    has_car: bool = True
    
    # Home location (zone_id where they live)
    home_zone: int = 1
    work_zone: int = 5

def create_mock_zone_graph():
    """Create realistic zone graph with 8 zones"""
    
    # Zone types and their characteristics
    zones_data = {
        1: {  # Sarah's home - Residential suburb
            "name": "Riverside Apartments",
            "type": "residential_medium",
            "population": 2500,
            "job_opportunities": 50,
            "retail_accessibility": 0.3,
            "transit_accessibility": 0.6,
            "attractiveness": 0.7,
            "coordinates": (0, 0)  # Reference point
        },
        2: {  # Neighborhood shopping
            "name": "Local Shopping Plaza", 
            "type": "retail_local",
            "population": 200,
            "job_opportunities": 300,
            "retail_accessibility": 0.9,
            "transit_accessibility": 0.7,
            "attractiveness": 0.6,
            "coordinates": (1, 0)
        },
        3: {  # Dense residential area
            "name": "Downtown Residential",
            "type": "residential_high", 
            "population": 8000,
            "job_opportunities": 100,
            "retail_accessibility": 0.8,
            "transit_accessibility": 0.9,
            "attractiveness": 0.8,
            "coordinates": (2, 0)
        },
        4: {  # Entertainment district
            "name": "Entertainment District",
            "type": "entertainment",
            "population": 500,
            "job_opportunities": 800,
            "retail_accessibility": 0.8,
            "transit_accessibility": 0.8,
            "attractiveness": 0.9,
            "coordinates": (2, 1)
        },
        5: {  # Sarah's workplace - Business district
            "name": "Tech Business Park",
            "type": "commercial_office",
            "population": 100,
            "job_opportunities": 5000,
            "retail_accessibility": 0.4,
            "transit_accessibility": 0.7,
            "attractiveness": 0.5,
            "coordinates": (3, 1)
        },
        6: {  # Major shopping center
            "name": "Grand Mall",
            "type": "retail_major",
            "population": 50,
            "job_opportunities": 1500,
            "retail_accessibility": 1.0,
            "transit_accessibility": 0.8,
            "attractiveness": 0.8,
            "coordinates": (3, 0)
        },
        7: {  # Gym/fitness area
            "name": "Fitness Complex",
            "type": "recreation",
            "population": 20,
            "job_opportunities": 200,
            "retail_accessibility": 0.2,
            "transit_accessibility": 0.5,
            "attractiveness": 0.7,
            "coordinates": (1, 1)
        },
        8: {  # Park/outdoor space
            "name": "Central Park",
            "type": "park",
            "population": 0,
            "job_opportunities": 50,
            "retail_accessibility": 0.1,
            "transit_accessibility": 0.4,
            "attractiveness": 0.9,
            "coordinates": (0, 1)
        }
    }
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for zone_id, data in zones_data.items():
        G.add_node(zone_id, **data)
    
    # Define edges (connections between adjacent zones)
    # Edge format: (from_zone, to_zone, {"distance": km, "travel_time": minutes})
    edges = [
        (1, 2, {"distance": 2.5, "travel_time": 8, "road_type": "local"}),
        (1, 7, {"distance": 3.2, "travel_time": 12, "road_type": "local"}), 
        (1, 8, {"distance": 1.8, "travel_time": 6, "road_type": "local"}),
        
        (2, 3, {"distance": 3.0, "travel_time": 10, "road_type": "arterial"}),
        (2, 6, {"distance": 4.5, "travel_time": 18, "road_type": "arterial"}),
        (2, 7, {"distance": 2.2, "travel_time": 7, "road_type": "local"}),
        
        (3, 4, {"distance": 1.5, "travel_time": 5, "road_type": "local"}),
        (3, 6, {"distance": 2.8, "travel_time": 12, "road_type": "arterial"}),
        
        (4, 5, {"distance": 2.0, "travel_time": 8, "road_type": "arterial"}),
        (4, 7, {"distance": 3.5, "travel_time": 15, "road_type": "local"}),
        
        (5, 6, {"distance": 1.2, "travel_time": 5, "road_type": "arterial"}),
        
        (7, 8, {"distance": 1.5, "travel_time": 5, "road_type": "local"}),
    ]
    
    G.add_edges_from(edges)
    
    return G, zones_data

def create_sarah_daily_pattern():
    """Sarah's realistic daily movement pattern"""
    
    # Time format: hours from midnight (0-24)
    daily_schedule = [
        {"time": 0.0, "zone": 1, "activity": "sleep", "description": "At home sleeping"},
        {"time": 7.0, "zone": 1, "activity": "morning_routine", "description": "Wake up, breakfast"},
        {"time": 7.5, "zone": 1, "activity": "prepare_commute", "description": "Getting ready to leave"},
        
        # Morning commute: Home → Work (via shopping plaza)
        {"time": 8.0, "zone": 1, "activity": "start_commute", "description": "Leaving home"},
        {"time": 8.13, "zone": 2, "activity": "transit", "description": "Passing through shopping area"},
        {"time": 8.27, "zone": 3, "activity": "transit", "description": "Through downtown residential"},
        {"time": 8.35, "zone": 4, "activity": "transit", "description": "Through entertainment district"},
        {"time": 8.45, "zone": 5, "activity": "arrive_work", "description": "Arriving at work"},
        
        # Work day
        {"time": 9.0, "zone": 5, "activity": "work", "description": "Morning work"},
        
        # Lunch break: Work → Mall → Work  
        {"time": 12.0, "zone": 5, "activity": "lunch_start", "description": "Leaving for lunch"},
        {"time": 12.08, "zone": 6, "activity": "lunch", "description": "Lunch at mall"},
        {"time": 13.0, "zone": 6, "activity": "lunch_end", "description": "Finishing lunch"},
        {"time": 13.08, "zone": 5, "activity": "work", "description": "Back to work"},
        
        # Afternoon work
        {"time": 17.0, "zone": 5, "activity": "end_work", "description": "Leaving work"},
        
        # Evening: Work → Gym → Home
        {"time": 17.15, "zone": 4, "activity": "transit", "description": "Through entertainment district"},
        {"time": 17.4, "zone": 7, "activity": "gym", "description": "Evening workout"},
        {"time": 19.0, "zone": 7, "activity": "gym_end", "description": "Leaving gym"},
        {"time": 19.08, "zone": 2, "activity": "transit", "description": "Passing shopping area"},
        {"time": 19.17, "zone": 1, "activity": "arrive_home", "description": "Back home"},
        
        # Evening at home
        {"time": 19.5, "zone": 1, "activity": "dinner", "description": "Dinner at home"},
        {"time": 21.0, "zone": 1, "activity": "evening", "description": "Relaxing at home"},
        {"time": 23.0, "zone": 1, "activity": "sleep", "description": "Going to sleep"},
    ]
    
    return daily_schedule


def create_training_data(person, schedule, zone_graph):
    """Convert schedule to ODE training format"""
    
    # Extract zone observations and times
    times = torch.tensor([event["time"] for event in schedule], dtype=torch.float32)
    zones = torch.tensor([event["zone"] - 1 for event in schedule], dtype=torch.long)  # Convert to 0-indexed
    
    # Person attributes as feature vector
    person_attrs = torch.tensor([
        person.age / 100.0,  # Normalize age
        person.income / 100000.0,  # Normalize income
        1.0 if person.employment_status == "full_time" else 0.0,
        1.0 if person.commute_preference == "car" else 0.0,
        person.activity_flexibility,
        person.social_tendency,
        person.household_size / 10.0,  # Normalize
        1.0 if person.has_car else 0.0
    ], dtype=torch.float32)
    
    # Zone graph as PyTorch Geometric format - convert to 0-indexed
    edge_list = [(u-1, v-1) for u, v in zone_graph.edges()]  # Convert to 0-indexed
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Zone features - reorder by zone_id to match 0-indexing
    zone_features = []
    for zone_id in sorted(zone_graph.nodes()):
        zone_data = zone_graph.nodes[zone_id]
        features = [
            zone_data["population"] / 10000.0,  # Normalize
            zone_data["job_opportunities"] / 5000.0,
            zone_data["retail_accessibility"],
            zone_data["transit_accessibility"], 
            zone_data["attractiveness"],
            zone_data["coordinates"][0] / 5.0,  # Normalize coordinates
            zone_data["coordinates"][1] / 5.0,
        ]
        zone_features.append(features)
    
    zone_features = torch.tensor(zone_features, dtype=torch.float32)
    
    # Edge features (travel times and distances) - reorder to match 0-indexed edges
    edge_features = []
    for u, v in edge_list:  # Use the 0-indexed edge list
        # Get original edge data using 1-indexed IDs
        edge_data = zone_graph.edges[(u+1, v+1)]
        features = [
            edge_data["distance"] / 10.0,  # Normalize distance
            edge_data["travel_time"] / 60.0,  # Normalize travel time
        ]
        edge_features.append(features)
    
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    
    return {
        "person_attrs": person_attrs,
        "times": times,
        "zone_observations": zones,  # Now 0-indexed
        "zone_features": zone_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "num_zones": len(zone_graph.nodes())
    }

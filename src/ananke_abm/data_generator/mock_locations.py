import networkx as nx
import torch
import numpy as np

def create_distance_matrix(zones_data):
    """
    Computes a pairwise distance matrix between all zones based on their coordinates.
    
    Args:
        zones_data (dict): Dictionary with zone information, including coordinates.
        
    Returns:
        torch.Tensor: A square matrix where element (i, j) is the Euclidean
                      distance between zone i+1 and zone j+1.
    """
    num_zones = len(zones_data)
    # Note: zone_ids are 1-based, so we create a 0-indexed array
    coords = np.array([zones_data[i+1]['coordinates'] for i in range(num_zones)])
    
    # Calculate pairwise squared Euclidean distances
    # (x1-x2)^2 + (y1-y2)^2
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    return torch.from_numpy(np.sqrt(dist_sq)).float()

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
    
    distance_matrix = create_distance_matrix(zones_data)
    
    return G, zones_data, distance_matrix
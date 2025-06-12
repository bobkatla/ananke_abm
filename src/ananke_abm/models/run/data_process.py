"""Process the data for the model"""

import pandas as pd
import numpy as np
from ananke_abm.data_generator.load_data import load_mobility_data, get_zone_adjacency_matrix
from ananke_abm.models.gnn_embed.HomoGraph import HomoGraph

def get_graphs(trajectories_dict, people_df, zones_df, adjacency_matrix):
    """Get the graphs for the model"""
    
    # Create locations graph
    # Prepare zone node features
    zone_node_features = zones_df.set_index('zone_id')[[
        'zone_type_retail', 'zone_type_residential', 'zone_type_office', 
        'zone_type_recreation', 'zone_type_transport', 'x_coord', 'y_coord',
        'population', 'job_opportunities', 'retail_accessibility', 
        'transit_accessibility', 'attractiveness'
    ]]
    zone_node_features.index.name = 'node_id'
    
    # Create zone edge features from adjacency matrix
    zone_edges = []
    num_zones = len(zones_df)
    for i in range(num_zones):
        for j in range(num_zones):
            if adjacency_matrix[i, j] == 1:
                zone_id_i = zones_df.iloc[i]['zone_id']
                zone_id_j = zones_df.iloc[j]['zone_id']
                # Calculate distance between zones as edge feature
                x1, y1 = zones_df.iloc[i]['x_coord'], zones_df.iloc[i]['y_coord']
                x2, y2 = zones_df.iloc[j]['x_coord'], zones_df.iloc[j]['y_coord']
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                zone_edges.append({
                    'edge_id': f"{zone_id_i}_{zone_id_j}",
                    'distance': distance,
                    'connected': 1.0
                })
    
    zone_edge_features = pd.DataFrame(zone_edges).set_index('edge_id')
    zone_edge_features.index.name = 'edge_id'
    
    locations_graph = HomoGraph(zone_node_features, zone_edge_features, bidirectional=False)
    
    # Create people graph
    # Prepare people node features
    people_node_features = people_df.set_index('person_id')[[
        'age', 'income', 'home_zone_id', 'work_zone_id'
    ]]
    people_node_features.index.name = 'node_id'
    
    # Create people edges (simple connections between all people for now)
    people_edges = []
    for i, person1 in people_df.iterrows():
        for j, person2 in people_df.iterrows():
            if person1['person_id'] != person2['person_id']:
                # Calculate similarity as edge feature
                age_diff = abs(person1['age'] - person2['age'])
                income_diff = abs(person1['income'] - person2['income'])
                same_home = 1.0 if person1['home_zone_id'] == person2['home_zone_id'] else 0.0
                same_work = 1.0 if person1['work_zone_id'] == person2['work_zone_id'] else 0.0
                
                people_edges.append({
                    'edge_id': f"{person1['person_id']}_{person2['person_id']}",
                    'age_similarity': 1.0 / (1.0 + age_diff/10.0),  # Normalized age similarity
                    'income_similarity': 1.0 / (1.0 + income_diff/10000.0),  # Normalized income similarity
                    'same_home_zone': same_home,
                    'same_work_zone': same_work
                })
    
    people_edge_features = pd.DataFrame(people_edges).set_index('edge_id')
    people_edge_features.index.name = 'edge_id'
    
    people_graph = HomoGraph(people_node_features, people_edge_features, bidirectional=False)
    
    print(f"🏘️  Created locations graph: {locations_graph}")
    print(f"👥 Created people graph: {people_graph}")
    
    return {
        'locations': locations_graph,
        'people': people_graph
    }

def process_mock_data():
    trajectories_dict, people_df, zones_df = load_mobility_data()
    adjacency_matrix = get_zone_adjacency_matrix()
    graphs = get_graphs(trajectories_dict, people_df, zones_df, adjacency_matrix)
    return graphs

if __name__ == "__main__":
    graphs = process_mock_data()
    print("\n📊 Graph Summary:")
    print(f"   Locations Graph Schema: {graphs['locations'].extract_schema()}")
    print(f"   People Graph Schema: {graphs['people'].extract_schema()}")
    
    # Demonstrate visualization
    print("\n🎨 Visualizing graphs...")
    
    # Visualize locations graph with different features FIRST
    print("Displaying locations graph...")
    graphs['locations'].visualize(
        title="Locations Graph - Zone Network",
        node_color_feature='population',
        node_size_feature='attractiveness',
        edge_color_feature='distance',
        layout='spring'
    )
    
    # Visualize people graph SECOND
    print("Displaying people graph...")
    graphs['people'].visualize(
        title="People Graph - Social Network",
        node_color_feature='age',
        node_size_feature='income',
        edge_color_feature='age_similarity',
        layout='circular'
    )
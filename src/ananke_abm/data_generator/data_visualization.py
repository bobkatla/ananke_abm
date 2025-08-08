"""Process the data for the model"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ananke_abm.data_generator.load_data import load_mobility_data, get_zone_adjacency_matrix

def get_graphs(people_df, zones_df, adjacency_matrix):
    """Create networkx graphs for zones and people."""
    # Create locations graph
    zone_graph = nx.Graph()
    for _, zone in zones_df.iterrows():
        zone_graph.add_node(zone['zone_id'], **zone.to_dict())

    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix)):
            if adjacency_matrix[i, j] == 1:
                zone_graph.add_edge(i + 1, j + 1)

    # Create people graph
    people_graph = nx.Graph()
    for _, person in people_df.iterrows():
        people_graph.add_node(person['person_id'], **person.to_dict())

    for i, p1 in people_df.iterrows():
        for j, p2 in people_df.iterrows():
            if p1['person_id'] != p2['person_id']:
                if p1['home_zone_id'] == p2['home_zone_id'] or p1['work_zone_id'] == p2['work_zone_id']:
                    people_graph.add_edge(p1['person_id'], p2['person_id'])
    
    return {'locations': zone_graph, 'people': people_graph}

def process_mock_data():
    trajectories_dict, people_df, zones_df = load_mobility_data()
    adjacency_matrix = get_zone_adjacency_matrix()
    graphs = get_graphs(people_df, zones_df, adjacency_matrix)
    
    # Load periods and snaps data
    data_dir = "data"
    periods_df = pd.read_csv(f"{data_dir}/periods.csv")
    snaps_df = pd.read_csv(f"{data_dir}/snaps.csv")
    
    return graphs, periods_df, snaps_df, zones_df

def visualize_zone_graph(zone_graph):
    """Visualizes the zone graph with population and attractiveness."""
    pos = {node: (data['x_coord'], data['y_coord']) for node, data in zone_graph.nodes(data=True)}
    node_sizes = [data['population'] / 5 for node, data in zone_graph.nodes(data=True)]
    node_colors = [data['attractiveness'] for node, data in zone_graph.nodes(data=True)]

    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = plt.cm.viridis
    
    nodes = nx.draw_networkx_nodes(zone_graph, pos, node_size=node_sizes, node_color=node_colors, cmap=cmap, ax=ax)
    nx.draw_networkx_edges(zone_graph, pos, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(zone_graph, pos, labels=nx.get_node_attributes(zone_graph, 'name'), font_size=8, ax=ax)
    
    # Add colorbar for attractiveness
    cbar = fig.colorbar(nodes, shrink=0.5, ax=ax)
    cbar.set_label('Attractiveness')

    # Add legend for node size (population)
    p_min, p_max = min(data['population'] for _, data in zone_graph.nodes(data=True)), max(data['population'] for _, data in zone_graph.nodes(data=True))
    handles = [plt.scatter([],[], s=p/5, label=f'{p:,}', color='skyblue') for p in [p_min, (p_min+p_max)//2, p_max]]
    ax.legend(handles=handles, title='Population', labelspacing=1.5, borderpad=1)

    ax.set_title('Zone Connectivity Graph')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    plt.show()

def visualize_people_graph(people_graph):
    """Visualizes the people graph with age and income."""
    pos = nx.spring_layout(people_graph, seed=42)
    node_sizes = [data['income'] / 50 for node, data in people_graph.nodes(data=True)]
    node_colors = [data['age'] for node, data in people_graph.nodes(data=True)]

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.coolwarm
    
    nodes = nx.draw_networkx_nodes(people_graph, pos, node_size=node_sizes, node_color=node_colors, cmap=cmap, ax=ax)
    nx.draw_networkx_edges(people_graph, pos, alpha=0.6, ax=ax)
    nx.draw_networkx_labels(people_graph, pos, labels=nx.get_node_attributes(people_graph, 'name'), font_size=10, ax=ax)

    # Add colorbar for age
    cbar = fig.colorbar(nodes, shrink=0.7, ax=ax)
    cbar.set_label('Age')

    # Add legend for node size (income)
    i_min, i_max = min(data['income'] for _, data in people_graph.nodes(data=True)), max(data['income'] for _, data in people_graph.nodes(data=True))
    handles = [plt.scatter([],[], s=i/50, label=f'${i:,.0f}', color='lightgrey') for i in [i_min, (i_min+i_max)//2, i_max]]
    ax.legend(handles=handles, title='Income', labelspacing=2, borderpad=1.2)
    
    ax.set_title('People Connectivity Graph')
    plt.show()

def visualize_agent_trajectories_over_time(snaps_df, periods_df, zones_df):
    """Visualizes agent trajectories with time on the x-axis and location on the y-axis."""
    fig, ax = plt.subplots(figsize=(18, 10))
    
    person_linestyles = {1: '-', 2: '--'}
    person_names = {1: 'Sarah', 2: 'Marcus'}
    
    # Define color and symbol mappings
    all_purposes = snaps_df['purpose'].unique()
    purpose_colors = {purpose: plt.cm.tab10(i) for i, purpose in enumerate(all_purposes)}
    
    mode_symbols = {
        "car": "C",
        "walk": "W",
        "bike": "B",
        "public_transit": "PT",
    }

    zone_name_to_id = {name: zid for zid, name in zones_df.set_index('zone_id')['name'].items()}

    for person_id, name in person_names.items():
        person_snaps = snaps_df[snaps_df['person_id'] == person_id].copy()
        person_snaps['location_id'] = person_snaps['location'].map(zone_name_to_id)
        
        ax.plot(person_snaps['timestamp'], person_snaps['location_id'], 
                 linestyle=person_linestyles[person_id],
                 color='black',
                 alpha=0.3,
                 label=f"{name}'s Trajectory")

        # Plot stay periods with purpose colors
        for _, period in periods_df[(periods_df['person_id'] == person_id) & (periods_df['type'] == 'stay')].iterrows():
            location_id = zone_name_to_id[period['location']]
            purpose = snaps_df[(snaps_df['person_id'] == person_id) & (snaps_df['timestamp'] == period['start_time'])]['purpose'].values[0]
            ax.fill_betweenx([location_id - 0.1, location_id + 0.1], period['start_time'], period['end_time'], 
                              color=purpose_colors[purpose], alpha=0.6)

        # Annotate travel periods with mode symbols
        for _, period in periods_df[(periods_df['person_id'] == person_id) & (periods_df['type'] == 'travel') & (periods_df['mode'] != 'stay')].iterrows():
            start_loc_id = zone_name_to_id[periods_df[(periods_df['person_id'] == person_id) & (periods_df['end_time'] == period['start_time'])]['location'].values[0]]
            end_loc_id = zone_name_to_id[periods_df[(periods_df['person_id'] == person_id) & (periods_df['start_time'] == period['end_time'])]['location'].values[0]]
            mid_time = (period['start_time'] + period['end_time']) / 2
            y_pos = (start_loc_id + end_loc_id) / 2
            
            symbol = mode_symbols.get(period['mode'].lower(), '?')
            ax.text(mid_time, y_pos, symbol, fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='circle,pad=0.2'))
    
    # Create legends
    purpose_patches = [plt.Rectangle((0, 0), 1, 1, color=color, label=purpose) for purpose, color in purpose_colors.items()]
    legend1 = ax.legend(handles=purpose_patches, title='Stay Purposes', loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.add_artist(legend1)

    # Manual legend for travel modes
    ax.text(1.02, 0.4, 'Travel Modes', transform=ax.transAxes, fontsize=10, weight='bold')
    y_offset = 0.35
    for mode, symbol in mode_symbols.items():
        ax.text(1.03, y_offset, f"{symbol} : {mode.replace('_', ' ').title()}", transform=ax.transAxes, fontsize=10, verticalalignment='top')
        y_offset -= 0.05

    ax.set_yticks(list(zone_name_to_id.values()))
    ax.set_yticklabels(list(zone_name_to_id.keys()))
    ax.set_xlabel('Time of Day (hours)')
    ax.set_ylabel('Location')
    ax.set_title('Agent Trajectories Over Time')
    ax.grid(True, axis='y')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

if __name__ == "__main__":
    _, people_df, zones_df = load_mobility_data()
    adjacency_matrix = get_zone_adjacency_matrix()
    graphs = get_graphs(people_df, zones_df, adjacency_matrix)
    
    periods_df = pd.read_csv("data/periods.csv")
    snaps_df = pd.read_csv("data/snaps.csv")

    print("\n🎨 Visualizing zone graph...")
    visualize_zone_graph(graphs['locations'])
    
    print("\n🎨 Visualizing people graph...")
    visualize_people_graph(graphs['people'])

    print("\n🎨 Visualizing agent trajectories...")
    visualize_agent_trajectories_over_time(snaps_df, periods_df, zones_df)
"""Common structure for a homogeneous graph that store some attributes along the GNN"""

import torch
import torch_geometric.data as Data
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Union, Optional

class HomoGraph:
    def __init__(self, node_features: pd.DataFrame, edge_features: pd.DataFrame, 
                 bidirectional: bool = True):
        """
        Initialize HomoGraph from pandas DataFrames
        
        Args:
            node_features: DataFrame with node_id as index and feature columns
            edge_features: DataFrame with edge_id as index (tuples of node pairs) and feature columns
            bidirectional: If True, automatically add reverse edges for undirected graphs
        """
        # Validate input format
        assert node_features.index.name == 'node_id', "node_features index must be named 'node_id'"
        assert edge_features.index.name == 'edge_id', "edge_features index must be named 'edge_id'"
        
        # Validate edge IDs format and node existence
        self._validate_edges(node_features.index, edge_features.index)
        
        # NOTE: data should have been normalized already
        self.map_idx_to_node_id = node_features.index.to_list()
        self.map_node_id_to_idx = {node_id: idx for idx, node_id in enumerate(self.map_idx_to_node_id)}
        
        # Store original DataFrames for visualization
        self.node_features_df = node_features
        self.edge_features_df = edge_features
        
        # Convert to tensors
        self.node_features = torch.tensor(node_features.values, dtype=torch.float32)
        
        # Process edges - handle bidirectionality
        if bidirectional:
            edge_features = self._make_bidirectional(edge_features)
            self.edge_features_df = edge_features
        
        self.map_idx_to_edge_id = edge_features.index.to_list()
        self.map_edge_id_to_idx = {edge_id: idx for idx, edge_id in enumerate(self.map_idx_to_edge_id)}
        self.edge_features = torch.tensor(edge_features.values, dtype=torch.float32)
        
        # Create edge_index in PyTorch Geometric format [2, num_edges]
        edge_list = []
        for edge_id in self.map_idx_to_edge_id:
            src_idx = self.map_node_id_to_idx[int(edge_id.split("_")[0])]
            dst_idx = self.map_node_id_to_idx[int(edge_id.split("_")[1])]
            edge_list.append([src_idx, dst_idx])
        
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create PyTorch Geometric Data object
        self.data = Data.Data(
            x=self.node_features, 
            edge_index=self.edge_index, 
            edge_attr=self.edge_features
        )
    
    def _validate_edges(self, node_ids: pd.Index, edge_ids: pd.Index):
        """Validate that edge IDs are proper tuples referencing existing nodes"""
        converted_edge_ids = [tuple(int(x) for x in edge_id.split("_")) for edge_id in edge_ids]
        for edge_id in converted_edge_ids:
            if not isinstance(edge_id, tuple) or len(edge_id) != 2:
                raise ValueError(f"edge_id must be tuple of (node1, node2), got {edge_id}")
            
            src_node, dst_node = edge_id
            if src_node not in node_ids:
                raise ValueError(f"Edge {edge_id} references non-existent source node {src_node}")
            if dst_node not in node_ids:
                raise ValueError(f"Edge {edge_id} references non-existent destination node {dst_node}")
    
    def _make_bidirectional(self, edge_features: pd.DataFrame) -> pd.DataFrame:
        """Add reverse edges to make graph bidirectional"""
        reverse_edges = []
        
        for edge_id, features in edge_features.iterrows():
            reverse_edge_id = f"{edge_id.split('_')[1]}_{edge_id.split('_')[0]}"
            
            # Only add if reverse doesn't already exist
            if reverse_edge_id not in edge_features.index:
                reverse_edges.append((reverse_edge_id, features))
        
        if reverse_edges:
            reverse_df = pd.DataFrame(
                [features for _, features in reverse_edges],
                index=pd.Index([edge_id for edge_id, _ in reverse_edges], name='edge_id')
            )
            edge_features = pd.concat([edge_features, reverse_df])
        
        return edge_features

    def get_data(self) -> Data.Data:
        """Get PyTorch Geometric Data object"""
        return self.data
    
    def extract_schema(self) -> Dict[str, int]:
        """Extract the schema from the graph as dict"""
        schema = {}
        schema['node_feature_dim'] = self.node_features.shape[1]
        schema['edge_feature_dim'] = self.edge_features.shape[1]
        schema['num_nodes'] = self.node_features.shape[0]
        schema['num_edges'] = self.edge_features.shape[0]
        return schema
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph"""
        return self.node_features.shape[0]
    
    @property
    def num_edges(self) -> int:
        """Number of edges in the graph"""
        return self.edge_features.shape[0]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'node_features': self.node_features.numpy(),
            'edge_features': self.edge_features.numpy(),
            'edge_index': self.edge_index.numpy(),
            'node_id_mapping': self.map_idx_to_node_id,
            'edge_id_mapping': self.map_idx_to_edge_id,
            'schema': self.extract_schema()
        }
    
    def __repr__(self) -> str:
        return f"HomoGraph(nodes={self.num_nodes}, edges={self.num_edges})"

    def get_neighbors(self, node_id) -> List:
        """Get neighboring node IDs for a given node"""
        node_idx = self.map_node_id_to_idx[node_id]
        neighbor_indices = self.edge_index[1][self.edge_index[0] == node_idx]
        return [self.map_idx_to_node_id[idx.item()] for idx in neighbor_indices]

    def get_edge_data(self, src_node_id, dst_node_id) -> torch.Tensor:
        """Get edge features between two nodes"""
        edge_id = (src_node_id, dst_node_id)
        if edge_id in self.map_edge_id_to_idx:
            edge_idx = self.map_edge_id_to_idx[edge_id]
            return self.edge_features[edge_idx]
        return None

    def visualize(self, 
                  figsize: Tuple[int, int] = (12, 8),
                  node_color_feature: Optional[str] = None,
                  edge_color_feature: Optional[str] = None,
                  node_size_feature: Optional[str] = None,
                  layout: str = 'spring',
                  show_labels: bool = True,
                  title: Optional[str] = None,
                  save_path: Optional[str] = None):
        """
        Visualize the graph using NetworkX and matplotlib
        
        Args:
            figsize: Figure size (width, height)
            node_color_feature: Node feature column name to use for coloring
            edge_color_feature: Edge feature column name to use for edge coloring
            node_size_feature: Node feature column name to use for node sizing
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'random', 'shell')
            show_labels: Whether to show node labels
            title: Plot title
            save_path: Path to save the figure (optional)
        """
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes with attributes
        for node_id in self.map_idx_to_node_id:
            node_attrs = self.node_features_df.loc[node_id].to_dict()
            G.add_node(node_id, **node_attrs)
        
        # Add edges with attributes
        for edge_id in self.map_idx_to_edge_id:
            src_node = int(edge_id.split("_")[0])
            dst_node = int(edge_id.split("_")[1])
            edge_attrs = self.edge_features_df.loc[edge_id].to_dict()
            G.add_edge(src_node, dst_node, **edge_attrs)
        
        # Clear any existing plots and setup fresh plot with white background
        plt.clf()
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'random':
            pos = nx.random_layout(G, seed=42)
        elif layout == 'shell':
            pos = nx.shell_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Node coloring
        if node_color_feature and node_color_feature in self.node_features_df.columns:
            node_colors = [self.node_features_df.loc[node, node_color_feature] for node in G.nodes()]
            node_cmap = plt.cm.viridis
        else:
            node_colors = 'lightblue'
            node_cmap = None
        
        # Node sizing
        if node_size_feature and node_size_feature in self.node_features_df.columns:
            raw_sizes = [self.node_features_df.loc[node, node_size_feature] for node in G.nodes()]
            # Normalize to reasonable range (100-1000)
            min_size, max_size = min(raw_sizes), max(raw_sizes)
            if max_size > min_size:
                node_sizes = [100 + 900 * (size - min_size) / (max_size - min_size) for size in raw_sizes]
            else:
                node_sizes = [300] * len(raw_sizes)
        else:
            node_sizes = 300
        
        # Edge coloring
        if edge_color_feature and edge_color_feature in self.edge_features_df.columns:
            edge_colors = []
            for edge in G.edges():
                edge_id = f"{edge[0]}_{edge[1]}"
                if edge_id in self.edge_features_df.index:
                    edge_colors.append(self.edge_features_df.loc[edge_id, edge_color_feature])
                else:
                    # Try reverse edge
                    reverse_edge_id = f"{edge[1]}_{edge[0]}"
                    if reverse_edge_id in self.edge_features_df.index:
                        edge_colors.append(self.edge_features_df.loc[reverse_edge_id, edge_color_feature])
                    else:
                        edge_colors.append(0.5)  # Default value
            edge_cmap = plt.cm.plasma
        else:
            edge_colors = 'gray'
            edge_cmap = None
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              cmap=node_cmap,
                              alpha=0.8,
                              ax=ax)
        
        nx.draw_networkx_edges(G, pos,
                              edge_color=edge_colors,
                              width=1.5,
                              alpha=0.6,
                              edge_cmap=edge_cmap,
                              ax=ax)
        
        if show_labels:
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        
        # Handle colorbars for both node and edge colors
        has_node_colors = node_color_feature and node_color_feature in self.node_features_df.columns
        has_edge_colors = edge_color_feature and edge_color_feature in self.edge_features_df.columns
        
        if has_node_colors and has_edge_colors:
            # Two colorbars side by side
            # Node colorbar on the right
            sm_nodes = plt.cm.ScalarMappable(cmap=node_cmap, 
                                           norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
            sm_nodes.set_array([])
            cbar_nodes = plt.colorbar(sm_nodes, ax=ax, shrink=0.6, pad=0.02, aspect=30)
            cbar_nodes.set_label(f'Node: {node_color_feature}', rotation=270, labelpad=20)
            
            # Edge colorbar on the far right
            sm_edges = plt.cm.ScalarMappable(cmap=edge_cmap, 
                                           norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)))
            sm_edges.set_array([])
            cbar_edges = plt.colorbar(sm_edges, ax=ax, shrink=0.6, pad=0.08, aspect=30)
            cbar_edges.set_label(f'Edge: {edge_color_feature}', rotation=270, labelpad=20)
            
        elif has_node_colors:
            # Only node colorbar
            sm_nodes = plt.cm.ScalarMappable(cmap=node_cmap, 
                                           norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
            sm_nodes.set_array([])
            cbar_nodes = plt.colorbar(sm_nodes, ax=ax, shrink=0.8)
            cbar_nodes.set_label(f'Node: {node_color_feature}', rotation=270, labelpad=20)
            
        elif has_edge_colors:
            # Only edge colorbar
            sm_edges = plt.cm.ScalarMappable(cmap=edge_cmap, 
                                           norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)))
            sm_edges.set_array([])
            cbar_edges = plt.colorbar(sm_edges, ax=ax, shrink=0.8)
            cbar_edges.set_label(f'Edge: {edge_color_feature}', rotation=270, labelpad=20)
        
        # Set title
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        else:
            ax.set_title(f"Graph Visualization ({self.num_nodes} nodes, {self.num_edges} edges)", 
                        fontsize=14, fontweight='bold', pad=20)
        
        # Add node size feature annotation
        if node_size_feature and node_size_feature in self.node_features_df.columns:
            ax.text(0.02, 0.98, f'Node size: {node_size_feature}', 
                   transform=ax.transAxes, fontsize=10, fontweight='bold',
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Graph visualization saved to: {save_path}")
        
        plt.show()
        
        return G

    @staticmethod
    def batch_graphs(graphs: List['HomoGraph']) -> 'Data.Batch':
        """Create batch from multiple HomoGraphs"""
        data_list = [graph.get_data() for graph in graphs]
        return Data.Batch.from_data_list(data_list)

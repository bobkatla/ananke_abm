"""Common structure for a homogeneous graph that store some attributes along the GNN"""

import torch
import torch_geometric.data as Data
import pandas as pd
from typing import Dict, Tuple, List, Union

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
        
        # Convert to tensors
        self.node_features = torch.tensor(node_features.values, dtype=torch.float32)
        
        # Process edges - handle bidirectionality
        if bidirectional:
            edge_features = self._make_bidirectional(edge_features)
        
        self.map_idx_to_edge_id = edge_features.index.to_list()
        self.map_edge_id_to_idx = {edge_id: idx for idx, edge_id in enumerate(self.map_idx_to_edge_id)}
        self.edge_features = torch.tensor(edge_features.values, dtype=torch.float32)
        
        # Create edge_index in PyTorch Geometric format [2, num_edges]
        edge_list = []
        for edge_id in self.map_idx_to_edge_id:
            src_idx = self.map_node_id_to_idx[edge_id[0]]
            dst_idx = self.map_node_id_to_idx[edge_id[1]]
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
        for edge_id in edge_ids:
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
            # Create reverse edge
            reverse_edge_id = (edge_id[1], edge_id[0])
            
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

    @staticmethod
    def batch_graphs(graphs: List['HomoGraph']) -> 'Data.Batch':
        """Create batch from multiple HomoGraphs"""
        data_list = [graph.get_data() for graph in graphs]
        return Data.Batch.from_data_list(data_list)

#!/usr/bin/env python3
"""
Rejection Sampling for Physics-Compliant Predictions

This module implements rejection sampling to guarantee that model predictions
comply with physics constraints by discarding invalid predictions and resampling
until valid ones are obtained.
"""

import torch
import time
import warnings
from typing import Tuple, Dict, List, Optional, Any
import numpy as np

class RejectionSampler:
    """
    Rejection sampler that ensures physics-compliant predictions.
    
    Uses rejection sampling to discard predictions that violate physics
    constraints and resample until valid predictions are obtained.
    """
    
    def __init__(self, adjacency_matrix: torch.Tensor, max_attempts: int = 1000,
                 timeout_seconds: float = 30.0, verbose: bool = True):
        """
        Initialize the rejection sampler.
        
        Args:
            adjacency_matrix: Physics constraint matrix (zones x zones)
            max_attempts: Maximum sampling attempts before giving up
            timeout_seconds: Maximum time to spend sampling
            verbose: Whether to print sampling statistics
        """
        self.adjacency_matrix = adjacency_matrix
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds
        self.verbose = verbose
        self.num_zones = adjacency_matrix.shape[0]
        
        # Statistics
        self.reset_stats()
    
    def reset_stats(self):
        """Reset sampling statistics"""
        self.stats = {
            'total_calls': 0,
            'total_attempts': 0,
            'successful_samples': 0,
            'failed_samples': 0,
            'average_attempts': 0.0,
            'total_time': 0.0,
            'timeout_rate': 0.0
        }
    
    def check_violations(self, predicted_path: torch.Tensor) -> Tuple[int, List[Tuple[int, int, int]]]:
        """
        Check physics violations in a predicted path.
        
        Args:
            predicted_path: Sequence of predicted zones [seq_len]
            
        Returns:
            violations: Number of violations
            violation_details: List of (step, from_zone, to_zone) tuples
        """
        violations = 0
        violation_details = []
        
        for i in range(len(predicted_path) - 1):
            curr_zone = predicted_path[i].item()
            next_zone = predicted_path[i+1].item()
            
            if self.adjacency_matrix[curr_zone, next_zone] == 0:
                violations += 1
                violation_details.append((i, curr_zone, next_zone))
        
        return violations, violation_details
    
    def sample_valid_prediction(self, model, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Sample a physics-compliant prediction using rejection sampling.
        
        Args:
            model: The neural model to sample from
            *args, **kwargs: Arguments to pass to the model
            
        Returns:
            valid_prediction: Physics-compliant prediction
            sampling_info: Statistics about the sampling process
        """
        start_time = time.time()
        self.stats['total_calls'] += 1
        
        for attempt in range(self.max_attempts):
            self.stats['total_attempts'] += 1
            
            # Check timeout
            if time.time() - start_time > self.timeout_seconds:
                self.stats['failed_samples'] += 1
                raise TimeoutError(f"Rejection sampling timed out after {self.timeout_seconds}s")
            
            # Get model prediction
            try:
                with torch.no_grad():
                    if hasattr(model, 'forward') and 'training' in model.forward.__code__.co_varnames:
                        zone_logits, trajectory = model(*args, **kwargs, training=False)
                    else:
                        zone_logits, trajectory = model(*args, **kwargs)
                    
                    predicted_zones = torch.argmax(zone_logits, dim=1)
                    
                    # Check for violations
                    violations, violation_details = self.check_violations(predicted_zones)
                    
                    if violations == 0:
                        # Success! Valid prediction found
                        self.stats['successful_samples'] += 1
                        sampling_time = time.time() - start_time
                        self.stats['total_time'] += sampling_time
                        
                        # Update average attempts
                        self.stats['average_attempts'] = (
                            self.stats['total_attempts'] / self.stats['total_calls']
                        )
                        
                        sampling_info = {
                            'attempts': attempt + 1,
                            'sampling_time': sampling_time,
                            'violations': violations,
                            'success': True
                        }
                        
                        if self.verbose and attempt > 0:
                            print(f"✅ Valid prediction found after {attempt + 1} attempts "
                                  f"({sampling_time:.3f}s)")
                        
                        return predicted_zones, sampling_info
                    
                    elif self.verbose and attempt % 100 == 99:
                        print(f"⏳ Attempt {attempt + 1}/{self.max_attempts}: "
                              f"{violations} violations, continuing...")
                        
            except Exception as e:
                if self.verbose:
                    print(f"❌ Model error on attempt {attempt + 1}: {e}")
                continue
        
        # Failed to find valid prediction
        self.stats['failed_samples'] += 1
        sampling_time = time.time() - start_time
        self.stats['total_time'] += sampling_time
        self.stats['timeout_rate'] = self.stats['failed_samples'] / self.stats['total_calls']
        
        raise RuntimeError(f"Failed to find valid prediction after {self.max_attempts} attempts")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sampling statistics"""
        return self.stats.copy()
    
    def print_stats(self):
        """Print detailed sampling statistics"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("🎯 REJECTION SAMPLING STATISTICS")
        print("="*50)
        print(f"Total calls: {stats['total_calls']}")
        print(f"Total attempts: {stats['total_attempts']}")
        print(f"Successful samples: {stats['successful_samples']}")
        print(f"Failed samples: {stats['failed_samples']}")
        print(f"Success rate: {stats['successful_samples'] / max(stats['total_calls'], 1):.1%}")
        print(f"Average attempts per success: {stats['average_attempts']:.1f}")
        print(f"Total sampling time: {stats['total_time']:.3f}s")
        print(f"Timeout rate: {stats['timeout_rate']:.1%}")

def physics_compliant_inference(model, person_attrs, times, zone_features, edge_index, 
                               adjacency_matrix, max_attempts: int = 1000,
                               timeout_seconds: float = 30.0, verbose: bool = True) -> Tuple[torch.Tensor, Dict]:
    """
    Convenience function for physics-compliant inference with rejection sampling.
    
    Args:
        model: Neural model
        person_attrs, times, zone_features, edge_index: Model inputs
        adjacency_matrix: Physics constraints
        max_attempts: Maximum sampling attempts
        timeout_seconds: Timeout for sampling
        verbose: Print progress
        
    Returns:
        valid_prediction: Physics-compliant prediction
        sampling_info: Sampling statistics
    """
    sampler = RejectionSampler(adjacency_matrix, max_attempts, timeout_seconds, verbose)
    
    try:
        prediction, info = sampler.sample_valid_prediction(
            model, person_attrs, times, zone_features, edge_index
        )
        return prediction, info
    except (TimeoutError, RuntimeError) as e:
        if verbose:
            print(f"🚨 Rejection sampling failed: {e}")
            sampler.print_stats()
        
        # Fallback: return best attempt with violation info
        with torch.no_grad():
            if hasattr(model, 'forward') and 'training' in model.forward.__code__.co_varnames:
                zone_logits, _ = model(person_attrs, times, zone_features, edge_index, training=False)
            else:
                zone_logits, _ = model(person_attrs, times, zone_features, edge_index)
            
            predicted_zones = torch.argmax(zone_logits, dim=1)
            violations, violation_details = sampler.check_violations(predicted_zones)
            
            fallback_info = {
                'attempts': max_attempts,
                'sampling_time': timeout_seconds,
                'violations': violations,
                'violation_details': violation_details,
                'success': False,
                'fallback': True
            }
            
            return predicted_zones, fallback_info

def batch_rejection_sampling(models_dict, person_attrs, times, zone_features, edge_index,
                            adjacency_matrix, max_attempts: int = 1000, 
                            timeout_seconds: float = 30.0, verbose: bool = True) -> Dict[str, Dict]:
    """
    Apply rejection sampling to multiple models.
    
    Args:
        models_dict: Dictionary of {model_name: model}
        person_attrs, times, zone_features, edge_index: Model inputs
        adjacency_matrix: Physics constraints
        max_attempts: Maximum attempts per model
        timeout_seconds: Timeout per model
        verbose: Print progress
        
    Returns:
        results: Dictionary of {model_name: {prediction, info}}
    """
    results = {}
    
    print(f"\n🎯 BATCH REJECTION SAMPLING ({len(models_dict)} models)")
    print("="*60)
    
    for model_name, model in models_dict.items():
        print(f"\n📊 Sampling {model_name}...")
        
        try:
            prediction, info = physics_compliant_inference(
                model, person_attrs, times, zone_features, edge_index,
                adjacency_matrix, max_attempts, timeout_seconds, verbose=False
            )
            
            results[model_name] = {
                'prediction': prediction,
                'info': info,
                'success': info['success']
            }
            
            if info['success']:
                print(f"   ✅ Success after {info['attempts']} attempts ({info['sampling_time']:.3f}s)")
            else:
                print(f"   ⚠️  Fallback used - {info['violations']} violations")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[model_name] = {
                'prediction': None,
                'info': {'success': False, 'error': str(e)},
                'success': False
            }
    
    # Summary statistics
    successful_models = sum(1 for r in results.values() if r['success'])
    print(f"\n📈 Batch Summary: {successful_models}/{len(models_dict)} models successful")
    
    return results

def create_adjacency_matrix(edge_index: torch.Tensor, num_zones: int) -> torch.Tensor:
    """
    Helper function to create adjacency matrix from edge index.
    
    Args:
        edge_index: Edge connectivity [2, num_edges]
        num_zones: Number of zones
        
    Returns:
        adjacency_matrix: Physics constraint matrix [num_zones, num_zones]
    """
    adjacency_matrix = torch.zeros(num_zones, num_zones)
    
    # Add edges
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        adjacency_matrix[u, v] = 1.0
        adjacency_matrix[v, u] = 1.0
    
    # Add self-loops (staying in same zone is always allowed)
    for i in range(num_zones):
        adjacency_matrix[i, i] = 1.0
    
    return adjacency_matrix 
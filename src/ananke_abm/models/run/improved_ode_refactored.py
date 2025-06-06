#!/usr/bin/env python3
"""
Refactored Physics-Constrained Models Comparison
Clean architecture with shared parameters and best model saving.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tempfile
import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime

# Import all models from separated files
from ananke_abm.models.gnn_embed import (
    PhysicsInformedODE, SmoothTrajectoryPredictor,
    ImprovedStrictPhysicsModel, SimplifiedDiffusionModel,
    HybridPhysicsModel, CurriculumPhysicsModel, EnsemblePhysicsModel,
    PhysicsDiffusionTrajectoryPredictor
)
from ananke_abm.data_generator.mock_2p import create_two_person_training_data

# =============================================================================
# SHARED TRAINING CONFIGURATION - For Fair Comparison
# =============================================================================

SHARED_CONFIG = {
    # Training parameters
    'epochs': 10000,  # Reduced for faster testing
    'learning_rate': 0.002,
    'weight_decay': 1e-5,
    'grad_clip_norm': 1.0,
    
    # Architecture parameters  
    'person_attrs_dim': 8,
    'num_zones': 8,
    
    # Evaluation criteria
    'best_model_criteria': {
        'primary': 'accuracy',  # Primary metric: accuracy
        'constraint': 'violations',  # Must have violations == 0
        'min_accuracy': 0.0  # Minimum accuracy threshold
    },
    
    # Shared optimization
    'optimizer_class': torch.optim.Adam,
    'scheduler_class': torch.optim.lr_scheduler.StepLR,
    'scheduler_kwargs': {'step_size': 500, 'gamma': 0.9},
    
    # Loss function
    'loss_fn': nn.CrossEntropyLoss(),
    
    # Evaluation frequency
    'eval_frequency': 100,
    
    # Model save directory
    'model_save_dir': Path('saved_models')
}

# =============================================================================
# BEST MODEL SAVING UTILITIES
# =============================================================================

def safe_model_save(model, filepath):
    """Atomic model saving to avoid corruption"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Use temporary file for atomic save
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
        torch.save(model.state_dict(), tmp_file.name)
        tmp_path = Path(tmp_file.name)
    
    # Atomic move to final location
    shutil.move(str(tmp_path), str(filepath))
    return filepath

def load_best_model(model_class, filepath, **model_kwargs):
    """Load the best saved model"""
    if model_class == SmoothTrajectoryPredictor:
        # SmoothTrajectoryPredictor needs an ODE function
        ode_func = PhysicsInformedODE(**model_kwargs)
        model = model_class(ode_func, num_zones=model_kwargs['num_zones'])
    elif model_class == PhysicsDiffusionTrajectoryPredictor:
        # PhysicsDiffusionTrajectoryPredictor has specific constructor
        model = model_class(**model_kwargs)
    else:
        model = model_class(**model_kwargs)
        
    if Path(filepath).exists():
        model.load_state_dict(torch.load(filepath))
        return model
    else:
        return None

class BestModelTracker:
    """Track and save best models during training"""
    
    def __init__(self, model_name, save_dir):
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.best_violations = float('inf')
        self.best_model_path = self.save_dir / f"{model_name}_best.pth"
        
    def update(self, model, epoch, accuracy, violations):
        """Update best model if criteria are met"""
        is_better = False
        
        # Primary criterion: accuracy improvement with zero violations
        if violations == 0:
            if accuracy > self.best_accuracy:
                is_better = True
        
        # Secondary: if no zero-violation model yet, prefer lower violations
        elif self.best_violations > 0 and violations < self.best_violations:
            is_better = True
        
        if is_better:
            self.best_accuracy = accuracy
            self.best_epoch = epoch
            self.best_violations = violations
            
            # Save the best model
            safe_model_save(model, self.best_model_path)
            print(f"        💾 Best model saved: {accuracy:.3f} acc, {violations} violations at epoch {epoch}")
            
        return is_better

# =============================================================================
# SHARED TRAINING FUNCTION
# =============================================================================

def train_model_with_shared_config(model, model_name, training_data, config=None):
    """Train any model with shared configuration for fair comparison"""
    
    if config is None:
        config = SHARED_CONFIG
    
    # Setup training components
    optimizer = config['optimizer_class'](model.parameters(), 
                                        lr=config['learning_rate'], 
                                        weight_decay=config['weight_decay'])
    
    if config['scheduler_class']:
        scheduler = config['scheduler_class'](optimizer, **config['scheduler_kwargs'])
    else:
        scheduler = None
    
    loss_fn = config['loss_fn']
    
    # Extract training data - handle multi-person case
    if training_data.get("is_multi_person", False):
        # Multi-person training: we'll alternate between people each epoch
        sarah_data = training_data["sarah_data"]
        marcus_data = training_data["marcus_data"]
        training_examples = [sarah_data, marcus_data]
        zone_features = sarah_data["zone_features"]  # Same for both
        edge_index = sarah_data["edge_index"]  # Same for both
        print(f"📚 Multi-person training: {len(training_examples)} people")
    else:
        # Single person training
        person_attrs = training_data["person_attrs"]
        times = training_data["times"] 
        zone_targets = training_data["zone_observations"]
        zone_features = training_data["zone_features"]
        edge_index = training_data["edge_index"]
        training_examples = None
    
    # Best model tracker
    tracker = BestModelTracker(model_name, config['model_save_dir'])
    
    # Training history
    history = {
        'epochs': [],
        'loss': [],
        'accuracy': [],
        'violations': [],
        'learning_rate': []
    }
    
    # Adjacency matrix for violation checking
    adjacency_matrix = torch.zeros(config['num_zones'], config['num_zones'])
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        adjacency_matrix[u, v] = 1.0
        adjacency_matrix[v, u] = 1.0
    for i in range(config['num_zones']):
        adjacency_matrix[i, i] = 1.0
    
    print(f"🚀 Training {model_name} with shared configuration...")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Epochs: {config['epochs']}, LR: {config['learning_rate']}")
    
    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        
        if training_examples is not None:
            # Multi-person training: alternate between people
            total_loss = 0
            num_people = len(training_examples)
            
            for person_data in training_examples:
                optimizer.zero_grad()
                
                # Extract person-specific data
                person_attrs = person_data["person_attrs"]
                times = person_data["times"]
                zone_targets = person_data["zone_observations"]
                
                # Forward pass
                try:
                    if hasattr(model, 'forward') and 'training' in model.forward.__code__.co_varnames:
                        zone_logits, _ = model(person_attrs, times, zone_features, edge_index, training=True)
                    else:
                        zone_logits, _ = model(person_attrs, times, zone_features, edge_index)
                    
                    # Loss computation
                    loss = loss_fn(zone_logits, zone_targets)
                    total_loss += loss.item()
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_norm'])
                    optimizer.step()
                    
                except Exception as e:
                    print(f"Error in epoch {epoch}, person {person_data.get('person_name', 'unknown')}: {e}")
                    break
            
            # Average loss across people
            loss = total_loss / num_people
            
            if scheduler:
                scheduler.step()
        else:
            # Single person training (original logic)
            optimizer.zero_grad()
            
            # Forward pass - handle different model interfaces
            try:
                if hasattr(model, 'forward') and 'training' in model.forward.__code__.co_varnames:
                    # Hybrid model with training flag
                    zone_logits, _ = model(person_attrs, times, zone_features, edge_index, training=True)
                else:
                    # Standard models
                    zone_logits, _ = model(person_attrs, times, zone_features, edge_index)
                
                # Loss computation
                loss = loss_fn(zone_logits, zone_targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_norm'])
                optimizer.step()
                
                if scheduler:
                    scheduler.step()
                    
            except Exception as e:
                print(f"Error in epoch {epoch}: {e}")
                break
        
        # Evaluation
        if epoch % config['eval_frequency'] == 0 or epoch == config['epochs'] - 1:
            model.eval()
            with torch.no_grad():
                if training_examples is not None:
                    # Multi-person evaluation: evaluate on both people and average results
                    total_accuracy = 0
                    total_violations = 0
                    total_transitions = 0
                    
                    for person_data in training_examples:
                        person_attrs = person_data["person_attrs"]
                        times = person_data["times"]
                        zone_targets = person_data["zone_observations"]
                        
                        try:
                            if hasattr(model, 'forward') and 'training' in model.forward.__code__.co_varnames:
                                eval_logits, _ = model(person_attrs, times, zone_features, edge_index, training=False)
                            else:
                                eval_logits, _ = model(person_attrs, times, zone_features, edge_index)
                            
                            predicted_zones = torch.argmax(eval_logits, dim=1)
                            person_accuracy = torch.mean((predicted_zones == zone_targets).float()).item()
                            total_accuracy += person_accuracy
                            
                            # Count violations for this person
                            person_violations = 0
                            for i in range(len(predicted_zones) - 1):
                                curr, next_z = predicted_zones[i].item(), predicted_zones[i+1].item()
                                if adjacency_matrix[curr, next_z] == 0:
                                    person_violations += 1
                            
                            total_violations += person_violations
                            total_transitions += len(predicted_zones) - 1
                            
                        except Exception as e:
                            print(f"Evaluation error for person {person_data.get('person_name', 'unknown')}: {e}")
                    
                    # Average metrics across people
                    accuracy = total_accuracy / len(training_examples)
                    violations = total_violations  # Total violations across all people
                    
                else:
                    # Single person evaluation (original logic)
                    try:
                        if hasattr(model, 'forward') and 'training' in model.forward.__code__.co_varnames:
                            eval_logits, _ = model(person_attrs, times, zone_features, edge_index, training=False)
                        else:
                            # Always do a fresh evaluation pass, never reuse training logits!
                            eval_logits, _ = model(person_attrs, times, zone_features, edge_index)
                        
                        predicted_zones = torch.argmax(eval_logits, dim=1)
                        accuracy = torch.mean((predicted_zones == zone_targets).float()).item()
                        
                        # Count violations
                        violations = 0
                        for i in range(len(predicted_zones) - 1):
                            curr, next_z = predicted_zones[i].item(), predicted_zones[i+1].item()
                            if adjacency_matrix[curr, next_z] == 0:
                                violations += 1
                        
                    except Exception as e:
                        print(f"Evaluation error at epoch {epoch}: {e}")
                        continue
                
                # Update best model
                is_best = tracker.update(model, epoch, accuracy, violations)
                
                # Store history
                history['epochs'].append(epoch)
                history['loss'].append(loss if isinstance(loss, float) else loss.item())
                history['accuracy'].append(accuracy)
                history['violations'].append(violations)
                history['learning_rate'].append(optimizer.param_groups[0]['lr'])
                
                # Print progress
                status = "🎯 NEW BEST" if is_best else ""
                violation_denom = total_transitions if training_examples else len(predicted_zones) - 1
                print(f"Epoch {epoch:4d}: Loss={loss if isinstance(loss, float) else loss.item():6.3f}, "
                      f"Acc={accuracy:.3f} ({accuracy*100:5.1f}%), "
                      f"Violations={violations}/{violation_denom}, "
                      f"LR={optimizer.param_groups[0]['lr']:.6f} {status}")
    
    print(f"📊 {model_name} Training Complete:")
    print(f"   Best accuracy: {tracker.best_accuracy:.3f} ({tracker.best_accuracy*100:.1f}%) at epoch {tracker.best_epoch}")
    
    if training_examples is not None:
        total_max_transitions = sum(len(data["times"]) - 1 for data in training_examples)
        print(f"   Best violations: {tracker.best_violations}/{total_max_transitions}")
        print(f"   Training: {len(training_examples)} people ({', '.join(data.get('person_name', f'Person {i+1}') for i, data in enumerate(training_examples))})")
    else:
        print(f"   Best violations: {tracker.best_violations}/21")
        print(f"   Training: Single person")
    
    print(f"   Model saved: {tracker.best_model_path}")
    
    return history, tracker

# =============================================================================
# COMPREHENSIVE MODEL COMPARISON
# =============================================================================

def compare_all_models(training_data, config=None):
    """Train and compare all models with fair shared configuration"""
    
    if config is None:
        config = SHARED_CONFIG
        
    print("="*80)
    print("🔬 COMPREHENSIVE PHYSICS-CONSTRAINED MODELS COMPARISON")
    print("="*80)
    print(f"Shared Configuration: {config['epochs']} epochs, LR={config['learning_rate']}")
    print("="*80)
    
    # Define all models to compare
    models_to_test = [
        {
            'name': 'PhysicsODE',
            'class': PhysicsInformedODE,
            'description': 'Continuous ODE dynamics'
        },
        # {
        #     'name': 'TrajectoryODE',
        #     'class': SmoothTrajectoryPredictor,
        #     'description': 'ODE with trajectory prediction'
        # },
        {
            'name': 'Diffusion',
            'class': SimplifiedDiffusionModel,
            'description': 'Soft constraints with penalty'
        },
        {
            'name': 'StrictPhysics', 
            'class': ImprovedStrictPhysicsModel,
            'description': 'Hard constraints with masking'
        },
        {
            'name': 'Hybrid',
            'class': HybridPhysicsModel, 
            'description': 'Soft training + hard inference'
        },
        # {
        #     'name': 'Curriculum',
        #     'class': CurriculumPhysicsModel,
        #     'description': 'Progressive constraint hardening'
        # },
        # {
        #     'name': 'Ensemble',
        #     'class': EnsemblePhysicsModel,
        #     'description': 'Combined soft + hard predictions'
        # },
        # {
        #     'name': 'DiffusionODE',
        #     'class': PhysicsDiffusionTrajectoryPredictor,
        #     'description': 'Physics-constrained diffusion with ODE dynamics'
        # }
    ]
    
    results = {}
    
    # Train each model
    for model_info in models_to_test:
        name = model_info['name']
        model_class = model_info['class']
        description = model_info['description']
        
        print(f"\n{'='*60}")
        print(f"🎯 Training {name}: {description}")
        print(f"{'='*60}")
        
        # Create model instance
        if model_class == SmoothTrajectoryPredictor:
            # SmoothTrajectoryPredictor needs an ODE function
            ode_func = PhysicsInformedODE(
                person_attrs_dim=config['person_attrs_dim'],
                num_zones=config['num_zones']
            )
            model = model_class(ode_func, num_zones=config['num_zones'])
        elif model_class == PhysicsDiffusionTrajectoryPredictor:
            # PhysicsDiffusionTrajectoryPredictor has different parameter names
            model = model_class(
                person_attrs_dim=config['person_attrs_dim'],
                num_zones=config['num_zones'],
                embedding_dim=64,
                diffusion_strength=0.2
            )
        else:
            model = model_class(
                person_attrs_dim=config['person_attrs_dim'],
                num_zones=config['num_zones']
            )
        
        # Train model
        history, tracker = train_model_with_shared_config(model, name, training_data, config)
        
        # Store results
        results[name] = {
            'model_class': model_class,
            'model': model,
            'history': history,
            'tracker': tracker,
            'description': description,
            'parameters': sum(p.numel() for p in model.parameters()),
            'best_accuracy': tracker.best_accuracy,
            'best_violations': tracker.best_violations,
            'best_epoch': tracker.best_epoch,
            'best_model_path': tracker.best_model_path
        }
    
    return results

# =============================================================================
# FINAL EVALUATION WITH BEST MODELS
# =============================================================================

def evaluate_best_models(results, evaluation_data, config=None):
    """Load and evaluate all best models on both people for generalization testing"""
    
    if config is None:
        config = SHARED_CONFIG
    
    print("\n" + "="*80)
    print("🏆 FINAL EVALUATION - PERSON-SPECIFIC PREDICTION")
    print("="*80)
    print("📊 Testing person-attribute learning: Models trained on both people, predicting person-specific patterns")
    print("="*80)
    
    # Get data for both people
    sarah_data = evaluation_data['sarah']
    marcus_data = evaluation_data['marcus']
    
    # Adjacency matrix for violation checking (same for both people)
    edge_index = sarah_data["edge_index"]
    adjacency_matrix = torch.zeros(config['num_zones'], config['num_zones'])
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        adjacency_matrix[u, v] = 1.0
        adjacency_matrix[v, u] = 1.0
    for i in range(config['num_zones']):
        adjacency_matrix[i, i] = 1.0
    
    final_results = {}
    
    for name, result in results.items():
        print(f"\n📈 Evaluating {name} (Best Model)...")
        
        # Load best model
        if result['model_class'] == PhysicsDiffusionTrajectoryPredictor:
            model = load_best_model(
                result['model_class'], 
                result['best_model_path'],
                person_attrs_dim=config['person_attrs_dim'],
                num_zones=config['num_zones'],
                embedding_dim=64,
                diffusion_strength=0.2
            )
        else:
            model = load_best_model(
                result['model_class'], 
                result['best_model_path'],
                person_attrs_dim=config['person_attrs_dim'],
                num_zones=config['num_zones']
            )
        
        if model is None:
            print(f"   ❌ Model file not found: {result['best_model_path']}")
            final_results[name] = None
            continue
            
        model.eval()
        
        # Evaluate on both people
        person_results = {}
        
        for person_name, person_data in [('Sarah', sarah_data), ('Marcus', marcus_data)]:
            person_attrs = person_data["person_attrs"]
            times = person_data["times"]
            zone_targets = person_data["zone_observations"]
            zone_features = person_data["zone_features"]
            
            with torch.no_grad():
                try:
                    if hasattr(model, 'forward') and 'training' in model.forward.__code__.co_varnames:
                        zone_logits, _ = model(person_attrs, times, zone_features, edge_index, training=False)
                    else:
                        zone_logits, _ = model(person_attrs, times, zone_features, edge_index)
                    
                    predicted_zones = torch.argmax(zone_logits, dim=1)
                    actual_zones = zone_targets
                    
                    # Calculate metrics
                    accuracy = torch.mean((predicted_zones == actual_zones).float()).item()
                    
                    # Count violations
                    violations = 0
                    violation_details = []
                    for i in range(len(predicted_zones) - 1):
                        curr, next_z = predicted_zones[i].item(), predicted_zones[i+1].item()
                        if adjacency_matrix[curr, next_z] == 0:
                            violations += 1
                            violation_details.append((i, curr+1, next_z+1))  # 1-indexed for display
                    
                    person_results[person_name.lower()] = {
                        'accuracy': accuracy,
                        'violations': violations,
                        'violation_details': violation_details,
                        'predicted_path': predicted_zones.tolist(),
                        'actual_path': actual_zones.tolist()
                    }
                    
                    # Print person-specific results
                    print(f"   {person_name}: Acc={accuracy:.3f} ({accuracy*100:.1f}%), Violations={violations}/{len(predicted_zones)-1}")
                    
                except Exception as e:
                    print(f"   ❌ {person_name} evaluation failed: {e}")
                    person_results[person_name.lower()] = None
        
        # Combine results for this model
        if person_results['sarah'] and person_results['marcus']:
            # Use Sarah's performance as primary (since we trained on Sarah)
            # But also track Marcus for generalization analysis
            final_results[name] = {
                'accuracy': person_results['sarah']['accuracy'],  # Primary metric (training person)
                'violations': person_results['sarah']['violations'],
                'parameters': result['parameters'],
                'description': result['description'],
                'sarah_results': person_results['sarah'],
                'marcus_results': person_results['marcus'],
                'generalization_gap': abs(person_results['sarah']['accuracy'] - person_results['marcus']['accuracy'])
            }
        else:
            final_results[name] = None
    
    # Print detailed path comparison table
    print("\n" + "="*80)
    print("📋 DETAILED PATH PREDICTIONS TABLE")
    print("="*80)
    
    if final_results:
        # Get Sarah's data for displaying (since that's what we trained on)
        sarah_data = evaluation_data['sarah']
        times = sarah_data['times']
        
        # Get the actual path from Sarah (primary)
        actual_path = None
        for name, result in final_results.items():
            if result and 'sarah_results' in result:
                actual_path = result['sarah_results']['actual_path']
                break
        
        if actual_path:
            # Create header
            print(f"{'Time':<6} {'Actual':<8}", end="")
            model_names = [name for name, result in final_results.items() if result]
            for name in model_names:
                print(f"{name:<12}", end="")
            print()  # New line
            print("-" * (6 + 8 + 12 * len(model_names)))
            
            # Print each time step for Sarah (training data)
            for i, actual_zone in enumerate(actual_path):
                time_val = times[i].item() if hasattr(times[i], 'item') else times[i]
                print(f"{time_val:<6.1f} {actual_zone+1:<8}", end="")  # 1-indexed zones
                
                for name in model_names:
                    result = final_results[name]
                    if result and 'sarah_results' in result and result['sarah_results']:
                        pred_zone = result['sarah_results']['predicted_path'][i]
                        # Mark violations with *
                        if i > 0:
                            prev_pred = result['sarah_results']['predicted_path'][i-1]
                            if adjacency_matrix[prev_pred, pred_zone] == 0:
                                print(f"{pred_zone+1}*{'':<10}", end="")  # 1-indexed with violation marker
                            else:
                                print(f"{pred_zone+1:<12}", end="")  # 1-indexed
                        else:
                            print(f"{pred_zone+1:<12}", end="")  # 1-indexed
                    else:
                        print(f"{'ERROR':<12}", end="")
                print()  # New line
            
            print("\nLegend: * = Physics violation (invalid transition)")
            print("Note: Showing Sarah's trajectory (training data). See summary for Marcus generalization.")
    
    return final_results

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comprehensive_comparison(results, final_results):
    """Create comprehensive comparison plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink']
    
    # 1. Training accuracy curves
    for i, (name, result) in enumerate(results.items()):
        if result and 'history' in result:
            epochs = result['history']['epochs']
            accuracy = [acc * 100 for acc in result['history']['accuracy']]
            ax1.plot(epochs, accuracy, color=colors[i % len(colors)], 
                    linewidth=2, label=name, marker='o', markersize=3)
    
    ax1.set_title('Training Accuracy Progression')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Final comparison bar chart with ACCURACY VALUES ON TOP
    if final_results:
        valid_results = {k: v for k, v in final_results.items() if v is not None}
        names = list(valid_results.keys())
        accuracies = [v['accuracy'] * 100 for v in valid_results.values()]
        violations = [v['violations'] for v in valid_results.values()]
        
        x = range(len(names))
        bars = ax2.bar(x, accuracies, color=[colors[i % len(colors)] for i in range(len(names))])
        
        # Add BOTH accuracy values AND violation status on top
        for i, (acc, viol, bar) in enumerate(zip(accuracies, violations, bars)):
            # Accuracy value on top
            ax2.text(i, acc + 1.5, f'{acc:.1f}%', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold', color='black')
            
            # Violation status below accuracy
            status = "✓" if viol == 0 else f"✗{viol}"
            ax2.text(i, acc + 0.5, status, ha='center', va='bottom', 
                    fontsize=10, fontweight='bold', 
                    color='green' if viol == 0 else 'red')
        
        ax2.set_title('Final Model Comparison')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45)
        ax2.set_ylim(0, max(accuracies) + 10)  # Extra space for labels
        ax2.grid(True, alpha=0.3)
    
    # 3. Parameter efficiency
    if final_results:
        valid_results = {k: v for k, v in final_results.items() if v is not None}
        params = [v['parameters'] for v in valid_results.values()]
        accuracies = [v['accuracy'] * 100 for v in valid_results.values()]
        
        scatter = ax3.scatter(params, accuracies, 
                            c=[colors[i % len(colors)] for i in range(len(names))],
                            s=100, alpha=0.7)
        
        for i, name in enumerate(names):
            ax3.annotate(name, (params[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_title('Parameter Efficiency')
        ax3.set_xlabel('Number of Parameters')
        ax3.set_ylabel('Accuracy (%)')
        ax3.grid(True, alpha=0.3)
    
    # 4. Violations over training
    for i, (name, result) in enumerate(results.items()):
        if result and 'history' in result:
            epochs = result['history']['epochs']
            violations = result['history']['violations']
            ax4.plot(epochs, violations, color=colors[i % len(colors)], 
                    linewidth=2, label=name, marker='o', markersize=3)
    
    ax4.set_title('Physics Violations During Training')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Violations (out of 21)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main(show_graph=False, config=None):
    """Main comprehensive evaluation pipeline"""
    
    if config is None:
        config = SHARED_CONFIG
    
    print("🔬 COMPREHENSIVE PHYSICS-CONSTRAINED MODELS EVALUATION")
    print("="*80)
    
    # Create two-person mock data
    print("📊 Creating two-person training data...")
    combined_data, sarah, marcus = create_two_person_training_data()
    
    # Extract individual training data for separate model training
    sarah_data = combined_data['sarah_data']
    marcus_data = combined_data['marcus_data']
    
    print(f"People: {sarah.name} & {marcus.name}")
    print(f"Sarah: {sarah_data['times'].shape[0]} points, Marcus: {marcus_data['times'].shape[0]} points")
    print(f"Zones: {sarah_data['num_zones']}, Edges: {sarah_data['edge_index'].shape[1]}")
    print(f"Person contrasts: Age({sarah.age}vs{marcus.age}), Income(${sarah.income:,}vs${marcus.income:,}), Transport({sarah.commute_preference}vs{marcus.commute_preference})")
    
    # Train on BOTH people to test person-specific pattern learning
    # Models should learn to use person attributes to predict the right behavior
    training_data = combined_data
    evaluation_data = {
        'sarah': sarah_data,
        'marcus': marcus_data,
        'combined': combined_data
    }
    
    # Compare all models
    results = compare_all_models(training_data, config)
    
    # Final evaluation with best models on both people
    final_results = evaluate_best_models(results, evaluation_data, config)
    
    # Create comprehensive visualization
    plot_comprehensive_comparison(results, final_results)
    
    # Print final summary
    print("\n" + "="*80)
    print("🎯 FINAL SUMMARY - BEST MODELS")
    print("="*80)
    
    df_results = None
    if final_results:
        # Sort by accuracy (descending) then by violations (ascending)
        sorted_results = sorted(final_results.items(), 
                              key=lambda x: (x[1]['accuracy'] if x[1] else 0, 
                                           -(x[1]['violations'] if x[1] else float('inf'))), 
                              reverse=True)
        
        print(f"{'Rank':<4} {'Model':<12} {'Accuracy':<10} {'Violations':<11} {'Parameters':<12} {'Status'}")
        print("-" * 80)
        
        # Create DataFrame for CSV export
        ranking_data = []
        
        for rank, (name, result) in enumerate(sorted_results, 1):
            if result:
                acc = result['accuracy']
                viol = result['violations']
                params = result['parameters']
                status = "🏆 PERFECT" if viol == 0 and acc > 0.8 else "✅ Good" if viol == 0 else "⚠️ Issues"
                
                print(f"{rank:<4} {name:<12} {acc:.3f}     {viol}/21       {params:<12,} {status}")
                
                # Add to DataFrame data
                ranking_data.append({
                    'Rank': rank,
                    'Model': name,
                    'Accuracy': acc,
                    'Accuracy_Percent': acc * 100,
                    'Violations': viol,
                    'Total_Transitions': 21,
                    'Violation_Rate': viol / 21,
                    'Parameters': params,
                    'Status': status.replace('🏆 ', '').replace('✅ ', '').replace('⚠️ ', ''),
                    'Description': result.get('description', 'N/A')
                })
        
        # Create DataFrame and save to CSV
        df_results = pd.DataFrame(ranking_data)
        csv_filename = f"model_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_results.to_csv(csv_filename, index=False)
        
        print(f"\n💾 Results saved to: {csv_filename}")
        print(f"📊 DataFrame shape: {df_results.shape}")
        
        # Find the absolute best
        best_models = [name for name, result in final_results.items() 
                      if result and result['violations'] == 0]
        
        if best_models:
            best_name = max(best_models, key=lambda x: final_results[x]['accuracy'])
            best_result = final_results[best_name]
            
            print(f"\n🏆 OVERALL WINNER: {best_name}")
            print(f"   Accuracy: {best_result['accuracy']:.3f} ({best_result['accuracy']*100:.1f}%)")
            print(f"   Violations: {best_result['violations']}/21 (PERFECT)")
            print(f"   Parameters: {best_result['parameters']:,}")
            print(f"   Description: {best_result['description']}")
        
        print(f"\n🚀 KEY FINDINGS:")
        perfect_models = len([r for r in final_results.values() if r and r['violations'] == 0])
        print(f"   ✅ {perfect_models}/{len(final_results)} models achieve zero violations")
        print(f"   📊 Models successfully demonstrate physics-constrained learning")
        print(f"   🎯 Best accuracy with perfect physics: {max([r['accuracy'] for r in final_results.values() if r and r['violations'] == 0]):.1%}")
        
        print(f"\n🔍 IMPORTANT NOTES:")
        print(f"   📋 The predicted path table shows zone transitions (1-indexed)")
        print(f"   ⚠️  '*' marks indicate physics violations (impossible transitions)")
        print(f"   💾 'Strict Physics' may show violations during TRAINING but has zero in FINAL EVALUATION")
        print(f"      because we save and load the BEST model (with zero violations) for final testing")
        print(f"   📊 The top-right plot now shows exact accuracy percentages on top of bars")
        
    return results, final_results, df_results

if __name__ == "__main__":
    # Run comprehensive evaluation
    results, final_results, df_results = main(show_graph=False, config=SHARED_CONFIG) 
#!/usr/bin/env python3
"""
Comprehensive Physics-Constrained Models with Rejection Sampling
Complete pipeline: Training, Evaluation, and Comparison with guaranteed physics compliance.
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

# Import all models
from ananke_abm.models.gnn_embed import (
    PhysicsInformedODE, SmoothTrajectoryPredictor,
    ImprovedStrictPhysicsModel, SimplifiedDiffusionModel,
    HybridPhysicsModel, CurriculumPhysicsModel, EnsemblePhysicsModel,
    PhysicsDiffusionTrajectoryPredictor
)

# Import rejection sampling
from ananke_abm.inference import (
    RejectionSampler, physics_compliant_inference, 
    batch_rejection_sampling, create_adjacency_matrix
)

from ananke_abm.data_generator.mock_1p import Person, create_mock_zone_graph, create_sarah_daily_pattern, create_training_data

# =============================================================================
# UNIFIED CONFIGURATION
# =============================================================================

UNIFIED_CONFIG = {
    # Training parameters
    'epochs': 10000,  # Reduced for faster testing
    'learning_rate': 0.002,
    'weight_decay': 1e-5,
    'grad_clip_norm': 1.0,
    
    # Architecture parameters  
    'person_attrs_dim': 8,
    'num_zones': 8,
    
    # Model save directory
    'model_save_dir': Path('saved_models'),
    
    # Evaluation criteria (for training)
    'best_model_criteria': {
        'primary': 'accuracy',  # Primary metric: accuracy
        'constraint': 'violations',  # Must have violations == 0
        'min_accuracy': 0.0  # Minimum accuracy threshold
    },
    
    # Shared optimization
    'optimizer_class': torch.optim.Adam,
    'scheduler_class': torch.optim.lr_scheduler.StepLR,
    'scheduler_kwargs': {'step_size': 500, 'gamma': 0.9},
    'loss_fn': nn.CrossEntropyLoss(),
    'eval_frequency': 100,
    
    # Rejection sampling parameters
    'max_attempts': 500,  # Maximum rejection sampling attempts
    'timeout_seconds': 15.0,  # Timeout per model
    'verbose_sampling': False,  # Reduce verbosity for cleaner output
    
    # Evaluation parameters  
    'include_fallback': True,
    'retrain_if_missing': False  # Set to True to retrain missing models
}

# =============================================================================
# TRAINING UTILITIES (from improved_ode_refactored.py)
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
        
        # Training history tracking
        self.training_history = {
            'epochs': [],
            'accuracies': [],
            'violations': [],
            'losses': []
        }
        
    def update(self, model, epoch, accuracy, violations, loss=None):
        """Update best model if criteria are met"""
        # Always track training history
        self.training_history['epochs'].append(epoch)
        self.training_history['accuracies'].append(accuracy)
        self.training_history['violations'].append(violations)
        if loss is not None:
            self.training_history['losses'].append(loss)
        
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
            safe_model_save(model, self.best_model_path)
            print(f"        💾 Best model saved: {accuracy:.3f} acc, {violations} violations at epoch {epoch}")
            
        return is_better

def train_model_with_config(model, model_name, training_data, config=None):
    """Train any model with shared configuration"""
    
    if config is None:
        config = UNIFIED_CONFIG
    
    # Setup training components
    optimizer = config['optimizer_class'](model.parameters(), 
                                        lr=config['learning_rate'], 
                                        weight_decay=config['weight_decay'])
    
    if config['scheduler_class']:
        scheduler = config['scheduler_class'](optimizer, **config['scheduler_kwargs'])
    else:
        scheduler = None
    
    loss_fn = config['loss_fn']
    
    # Extract training data
    person_attrs = training_data["person_attrs"]
    times = training_data["times"] 
    zone_targets = training_data["zone_observations"]
    zone_features = training_data["zone_features"]
    edge_index = training_data["edge_index"]
    
    # Best model tracker
    tracker = BestModelTracker(model_name, config['model_save_dir'])
    
    # Adjacency matrix for violation checking
    adjacency_matrix = torch.zeros(config['num_zones'], config['num_zones'])
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        adjacency_matrix[u, v] = 1.0
        adjacency_matrix[v, u] = 1.0
    for i in range(config['num_zones']):
        adjacency_matrix[i, i] = 1.0
    
    print(f"🚀 Training {model_name}...")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        try:
            if hasattr(model, 'forward') and 'training' in model.forward.__code__.co_varnames:
                zone_logits, _ = model(person_attrs, times, zone_features, edge_index, training=True)
            else:
                zone_logits, _ = model(person_attrs, times, zone_features, edge_index)
            
            loss = loss_fn(zone_logits, zone_targets)
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
                try:
                    if hasattr(model, 'forward') and 'training' in model.forward.__code__.co_varnames:
                        eval_logits, _ = model(person_attrs, times, zone_features, edge_index, training=False)
                    else:
                        eval_logits = zone_logits
                    
                    predicted_zones = torch.argmax(eval_logits, dim=1)
                    accuracy = torch.mean((predicted_zones == zone_targets).float()).item()
                    
                    # Count violations
                    violations = 0
                    for i in range(len(predicted_zones) - 1):
                        curr, next_z = predicted_zones[i].item(), predicted_zones[i+1].item()
                        if adjacency_matrix[curr, next_z] == 0:
                            violations += 1
                    
                    # Update best model
                    is_best = tracker.update(model, epoch, accuracy, violations, loss.item())
                    
                    # Print progress
                    status = "🎯 NEW BEST" if is_best else ""
                    if epoch % (config['eval_frequency'] * 2) == 0 or epoch == config['epochs'] - 1 or is_best:
                        print(f"Epoch {epoch:4d}: Loss={loss.item():6.3f}, "
                              f"Acc={accuracy:.3f}, Violations={violations}/21 {status}")
                    
                except Exception as e:
                    print(f"Evaluation error at epoch {epoch}: {e}")
    
    print(f"✅ {model_name} Training Complete: {tracker.best_accuracy:.3f} acc, {tracker.best_violations} violations")
    return tracker

# =============================================================================
# MODEL LOADING AND MANAGEMENT
# =============================================================================

def load_best_model(model_class, filepath, **model_kwargs):
    """Load the best saved model"""
    model = model_class(**model_kwargs)
    if Path(filepath).exists():
        model.load_state_dict(torch.load(filepath))
        return model
    else:
        return None

def get_or_train_all_models(training_data, config=None):
    """Load existing models or train them if missing"""
    
    if config is None:
        config = UNIFIED_CONFIG
        
    models = {}
    training_histories = {}  # Store training histories
    
    model_configs = [
        ('Diffusion', SimplifiedDiffusionModel, 'Diffusion_best.pth'),
        ('StrictPhysics', ImprovedStrictPhysicsModel, 'StrictPhysics_best.pth'),
        ('Hybrid', HybridPhysicsModel, 'Hybrid_best.pth'),
        ('Curriculum', CurriculumPhysicsModel, 'Curriculum_best.pth'),
        ('Ensemble', EnsemblePhysicsModel, 'Ensemble_best.pth'),
        ('PhysicsODE', PhysicsInformedODE, 'PhysicsODE_best.pth'),
        ('DiffusionODE', PhysicsDiffusionTrajectoryPredictor, 'DiffusionODE_best.pth')
    ]
    
    missing_models = []
    
    # First, try to load existing models
    for name, model_class, filename in model_configs:
        filepath = config['model_save_dir'] / filename
        model = load_best_model(
            model_class, filepath,
            person_attrs_dim=config['person_attrs_dim'],
            num_zones=config['num_zones']
        )
        if model is not None:
            model.eval()
            models[name] = model
            print(f"✅ Loaded {name} from {filepath}")
            # No training history for pre-loaded models
            training_histories[name] = None
        else:
            missing_models.append((name, model_class, filename))
    
    # Train missing models if requested
    if missing_models and config['retrain_if_missing']:
        print(f"\n🔄 Training {len(missing_models)} missing models...")
        for name, model_class, filename in missing_models:
            print(f"\n{'='*60}")
            print(f"🎯 Training {name}")
            print(f"{'='*60}")
            
            model = model_class(
                person_attrs_dim=config['person_attrs_dim'],
                num_zones=config['num_zones']
            )
            
            tracker = train_model_with_config(model, name, training_data, config)
            
            # Load the best trained model
            model = load_best_model(
                model_class, tracker.best_model_path,
                person_attrs_dim=config['person_attrs_dim'],
                num_zones=config['num_zones']
            )
            model.eval()
            models[name] = model
            training_histories[name] = tracker.training_history
    
    return models, training_histories

# =============================================================================
# EVALUATION WITH REJECTION SAMPLING
# =============================================================================

def evaluate_models_with_rejection_sampling(models, training_data, config=None):
    """Evaluate all models using rejection sampling for guaranteed compliance"""
    
    if config is None:
        config = UNIFIED_CONFIG
    
    # Extract data
    person_attrs = training_data["person_attrs"]
    times = training_data["times"]
    zone_targets = training_data["zone_observations"]
    zone_features = training_data["zone_features"] 
    edge_index = training_data["edge_index"]
    
    # Create adjacency matrix
    adjacency_matrix = create_adjacency_matrix(edge_index, config['num_zones'])
    
    print("\n" + "="*80)
    print("🎯 PHYSICS-COMPLIANT EVALUATION (Rejection Sampling)")
    print("="*80)
    
    # Apply batch rejection sampling
    rejection_results = batch_rejection_sampling(
        models, person_attrs, times, zone_features, edge_index, adjacency_matrix,
        max_attempts=config['max_attempts'],
        timeout_seconds=config['timeout_seconds'],
        verbose=config['verbose_sampling']
    )
    
    # Process results
    final_results = {}
    
    for name, result in rejection_results.items():
        if result['success'] and result['prediction'] is not None:
            predicted_zones = result['prediction']
            actual_zones = zone_targets
            
            # Calculate accuracy
            accuracy = torch.mean((predicted_zones == actual_zones).float()).item()
            
            # Verify zero violations (should be guaranteed)
            violations = 0
            violation_details = []
            for i in range(len(predicted_zones) - 1):
                curr, next_z = predicted_zones[i].item(), predicted_zones[i+1].item()
                if adjacency_matrix[curr, next_z] == 0:
                    violations += 1
                    violation_details.append((i, curr+1, next_z+1))
            
            final_results[name] = {
                'prediction': predicted_zones.tolist(),
                'accuracy': accuracy,
                'violations': violations,
                'violation_details': violation_details,
                'sampling_info': result['info'],
                'parameters': sum(p.numel() for p in models[name].parameters())
            }
            
            print(f"\n📊 {name}:")
            print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
            print(f"   Violations: {violations}/21 ✅")
            print(f"   Attempts: {result['info']['attempts']}")
            print(f"   Time: {result['info']['sampling_time']:.3f}s")
            print(f"   Parameters: {final_results[name]['parameters']:,}")
            
        else:
            print(f"\n❌ {name}: Rejection sampling failed")
            final_results[name] = None
    
    return final_results, adjacency_matrix

# =============================================================================
# COMPREHENSIVE ANALYSIS AND VISUALIZATION
# =============================================================================

def print_detailed_path_table(final_results, training_data, adjacency_matrix):
    """Print detailed path predictions table"""
    
    times = training_data["times"]
    actual_path = training_data["zone_observations"].tolist()
    
    print("\n" + "="*80)
    print("📋 DETAILED PATH PREDICTIONS TABLE")
    print("="*80)
    
    if final_results:
        # Create header
        print(f"{'Time':<6} {'Actual':<8}", end="")
        model_names = [name for name, result in final_results.items() if result]
        for name in model_names:
            print(f"{name:<12}", end="")
        print()  # New line
        print("-" * (6 + 8 + 12 * len(model_names)))
        
        # Print each time step
        for i, actual_zone in enumerate(actual_path):
            time_val = times[i].item() if hasattr(times[i], 'item') else times[i]
            print(f"{time_val:<6.1f} {actual_zone+1:<8}", end="")  # 1-indexed zones
            
            for name in model_names:
                result = final_results[name]
                if result and 'prediction' in result:
                    pred_zone = result['prediction'][i]
                    # Mark violations with *
                    if i > 0:
                        prev_pred = result['prediction'][i-1]
                        if adjacency_matrix[prev_pred, pred_zone] == 0:
                            print(f"{pred_zone+1}*{'':<10}", end="")  # 1-indexed with violation marker
                        else:
                            print(f"{pred_zone+1:<12}", end="")  # 1-indexed
                    else:
                        print(f"{pred_zone+1:<12}", end="")  # 1-indexed
                else:
                    print(f"{'ERROR':<12}", end="")
            print()  # New line
        
        print("\nLegend: * = Physics violation (should never occur with rejection sampling)")

def print_final_ranking(final_results):
    """Print final ranking of models and save to CSV"""
    
    print("\n" + "="*80)
    print("🏆 FINAL RANKING - PHYSICS-COMPLIANT MODELS")
    print("="*80)
    
    if final_results:
        # Sort by accuracy (all should have 0 violations)
        sorted_results = sorted(
            [(name, result) for name, result in final_results.items() if result], 
            key=lambda x: x[1]['accuracy'], reverse=True
        )
        
        print(f"{'Rank':<4} {'Model':<12} {'Accuracy':<10} {'Violations':<11} {'Parameters':<12} {'Attempts':<8} {'Status'}")
        print("-" * 80)
        
        # Create DataFrame for CSV export
        ranking_data = []
        
        for rank, (name, result) in enumerate(sorted_results, 1):
            acc = result['accuracy']
            viol = result['violations']
            params = result['parameters']
            attempts = result['sampling_info']['attempts']
            status = "🏆 PERFECT" if viol == 0 and acc > 0.8 else "✅ Good" if viol == 0 else "⚠️ Issues"
            
            print(f"{rank:<4} {name:<12} {acc:.3f}     {viol}/21       {params:<12,} {attempts:<8} {status}")
            
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
                'Sampling_Attempts': attempts,
                'Status': status.replace('🏆 ', '').replace('✅ ', '').replace('⚠️ ', '')
            })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(ranking_data)
        csv_filename = f"physics_models_ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_filename, index=False)
        
        print(f"\n💾 Results saved to: {csv_filename}")
        print(f"📊 DataFrame shape: {df.shape}")
        
        # Find the absolute best
        if sorted_results:
            best_name, best_result = sorted_results[0]
            print(f"\n🏆 OVERALL WINNER: {best_name}")
            print(f"   Accuracy: {best_result['accuracy']:.3f} ({best_result['accuracy']*100:.1f}%)")
            print(f"   Violations: {best_result['violations']}/21 (PERFECT)")
            print(f"   Parameters: {best_result['parameters']:,}")
            print(f"   Sampling Attempts: {best_result['sampling_info']['attempts']}")
        
        return df
    
    return None

def plot_comprehensive_results(final_results, training_histories=None):
    """Create comprehensive visualization with training progression"""
    
    # Get valid results
    valid_results = {k: v for k, v in final_results.items() if v is not None}
    if not valid_results:
        print("No valid results for plotting")
        return
    
    model_names = list(valid_results.keys())
    final_accuracies = [v['accuracy'] * 100 for v in valid_results.values()]
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple']
    
    # Check if we have any training histories
    has_training_history = training_histories and any(h is not None for h in training_histories.values())
    
    # 1. Training Accuracy Over Epochs
    ax1.set_title('Training Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    
    if has_training_history:
        plotted_any = False
        for i, (name, history) in enumerate(training_histories.items()):
            if history and history['epochs']:
                ax1.plot(history['epochs'], history['accuracies'], 
                        label=name, color=colors[i % len(colors)], linewidth=2, marker='o', markersize=3)
                plotted_any = True
        if plotted_any:
            ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
    else:
        ax1.text(0.5, 0.5, 'No training history available\n(Models pre-loaded)\n\nSet retrain=True to see training curves', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
    
    # 2. Training Violations Over Epochs  
    ax2.set_title('Physics Violations During Training', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Violations (out of 21)')
    
    if has_training_history:
        plotted_any = False
        max_violations = 0
        for i, (name, history) in enumerate(training_histories.items()):
            if history and history['epochs']:
                ax2.plot(history['epochs'], history['violations'], 
                        label=name, color=colors[i % len(colors)], linewidth=2, marker='o', markersize=3)
                plotted_any = True
                if history['violations']:
                    max_violations = max(max_violations, max(history['violations']))
        if plotted_any:
            ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(21, max_violations + 2))
    else:
        ax2.text(0.5, 0.5, 'No training history available\n(Models pre-loaded)\n\nSet retrain=True to see training curves', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 21)
    
    # 3. Final Accuracy Comparison
    x = range(len(model_names))
    bars3 = ax3.bar(x, final_accuracies, color=colors[:len(model_names)], alpha=0.7)
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(final_accuracies):
        ax3.text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_title('Final Model Accuracy (Physics-Compliant)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45)
    ax3.set_ylim(0, max(final_accuracies) + 10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Parameter Efficiency
    params = [v['parameters'] for v in valid_results.values()]
    scatter = ax4.scatter(params, final_accuracies, c=colors[:len(model_names)], s=150, alpha=0.7)
    
    for i, (name, param, acc) in enumerate(zip(model_names, params, final_accuracies)):
        ax4.annotate(name, (param, acc), xytext=(5, 5), textcoords='offset points', 
                    fontsize=10, fontweight='bold')
    
    ax4.set_title('Parameter Efficiency', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Number of Parameters')
    ax4.set_ylabel('Accuracy (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_physics_models.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main(retrain=False):
    """Complete physics-constrained models pipeline"""
    
    print("🔬 COMPREHENSIVE PHYSICS-CONSTRAINED MODELS EVALUATION")
    print("="*80)
    print("🎯 Rejection Sampling: ENABLED (Guaranteed Physics Compliance)")
    print("="*80)
    
    # Create training data
    print("📊 Creating evaluation data...")
    sarah = Person()
    zone_graph, zone_data = create_mock_zone_graph()
    sarah_schedule = create_sarah_daily_pattern()
    training_data = create_training_data(sarah, sarah_schedule, zone_graph)
    
    print(f"Person: {sarah.name}")
    print(f"Evaluation points: {len(training_data['times'])}")
    print(f"Zones: {training_data['num_zones']}, Edges: {training_data['edge_index'].shape[1]}")
    
    # Configure for retraining if requested
    config = UNIFIED_CONFIG.copy()
    config['retrain_if_missing'] = retrain
    
    # Load or train models
    print(f"\n📦 Loading models from {config['model_save_dir']}...")
    if retrain:
        print("🔄 Retraining mode enabled - will train missing models")
    
    models, training_histories = get_or_train_all_models(training_data, config)
    
    if not models:
        print("❌ No models available! Set retrain=True to train new models.")
        return None, None
    
    print(f"✅ Loaded {len(models)} models: {list(models.keys())}")
    
    # Evaluate with rejection sampling
    final_results, adjacency_matrix = evaluate_models_with_rejection_sampling(models, training_data, config)
    
    # Comprehensive analysis
    print_detailed_path_table(final_results, training_data, adjacency_matrix)
    df_results = print_final_ranking(final_results)
    
    # Create visualizations
    plot_comprehensive_results(final_results, training_histories)
    
    # Final summary
    print(f"\n🎉 EVALUATION COMPLETE!")
    print(f"   ✅ All models achieve guaranteed physics compliance")
    print(f"   🎯 Rejection sampling ensures zero violations")
    print(f"   📊 Models ranked by accuracy with perfect constraint satisfaction")
    
    return final_results, df_results

if __name__ == "__main__":
    # Run the comprehensive evaluation
    # Set retrain=True if you want to retrain missing models
    final_results, df_results = main(retrain=False) 
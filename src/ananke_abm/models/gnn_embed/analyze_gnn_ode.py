#!/usr/bin/env python3
"""
GNN-ODE Model Analysis and Visualization
Load saved models, evaluate performance, and create comprehensive analysis plots.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Import our model components
from .gnn_ode import GNNPhysicsODE, GNNODETrainer
from .HomoGraph import HomoGraph
from ..run.run_gnn_ode import (
    create_location_graph, 
    create_person_graph, 
    extract_trajectory_data,
    check_physics_violations,
    show_graph_connectivity
)

class GNNODEAnalyzer:
    """Comprehensive analysis of GNN-ODE model performance"""
    
    def __init__(self, model_dir: str = "saved_models", results_dir: str = "analysis_results"):
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.location_graph = create_location_graph()
        self.person_graph = create_person_graph()
        self.trajectories = extract_trajectory_data()
        
        # Initialize model
        self.model = GNNPhysicsODE(
            location_graph=self.location_graph,
            person_graph=self.person_graph,
            embedding_dim=64,
            num_gnn_layers=2
        )
        
        self.trainer = GNNODETrainer(self.model, save_dir=str(self.model_dir))
        
    def load_model(self, model_path: str = None):
        """Load a saved model"""
        if model_path is None:
            model_path = self.model_dir / "gnn_ode_best.pth"
        
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path))
            print(f"✅ Model loaded from: {model_path}")
            return True
        else:
            print(f"❌ Model not found: {model_path}")
            return False
    
    def evaluate_model(self) -> Dict:
        """Evaluate the loaded model"""
        print("🔍 Evaluating model...")
        results = self.trainer.evaluate(self.trajectories)
        
        # Add physics violation analysis
        total_violations, total_transitions = check_physics_violations(results, self.location_graph)
        
        # Calculate overall metrics
        overall_accuracy = np.mean([r['accuracy'] for r in results.values()])
        violation_rate = total_violations / total_transitions if total_transitions > 0 else 0
        
        # Add summary metrics
        results['summary'] = {
            'overall_accuracy': overall_accuracy,
            'total_violations': total_violations,
            'total_transitions': total_transitions,
            'violation_rate': violation_rate
        }
        
        return results
    
    def save_predictions_csv(self, results: Dict):
        """Save predictions as CSV files"""
        print("💾 Saving predictions to CSV...")
        
        for person_name, result in results.items():
            if person_name == 'summary':
                continue
                
            # Create detailed CSV for each person
            df = pd.DataFrame({
                'time': result['times'].numpy(),
                'observed_zone': result['observed_zones'].numpy(),
                'predicted_zone': result['predicted_zones'].numpy(),
                'correct': (result['predicted_zones'] == result['observed_zones']).numpy()
            })
            
            csv_path = self.results_dir / f"predictions_{person_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  📄 {person_name}: {csv_path}")
        
        # Save summary CSV
        summary_data = []
        for person_name, result in results.items():
            if person_name == 'summary':
                continue
            summary_data.append({
                'person': person_name,
                'accuracy': result['accuracy'],
                'num_predictions': len(result['times']),
                'correct_predictions': (result['predicted_zones'] == result['observed_zones']).sum().item()
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.results_dir / "evaluation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"  📄 Summary: {summary_path}")
    
    def load_training_curves(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training loss and learning rate curves"""
        loss_path = self.model_dir / "gnn_ode_training_losses.npy"
        lr_path = self.model_dir / "gnn_ode_learning_rates.npy"
        
        losses = np.load(loss_path) if loss_path.exists() else np.array([])
        lrs = np.load(lr_path) if lr_path.exists() else np.array([])
        
        return losses, lrs
    
    def plot_training_analysis(self):
        """Create comprehensive training analysis plots"""
        print("📊 Creating training analysis plots...")
        
        losses, lrs = self.load_training_curves()
        
        if len(losses) == 0:
            print("❌ No training data found!")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GNN-ODE Training Analysis', fontsize=16)
        
        # 1. Loss curve
        axes[0, 0].plot(losses, 'b-', linewidth=2)
        axes[0, 0].set_title('Training Loss Over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # 2. Learning rate schedule
        if len(lrs) > 0:
            axes[0, 1].plot(lrs, 'r-', linewidth=2)
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_yscale('log')
        else:
            axes[0, 1].text(0.5, 0.5, 'No LR data', ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # 3. Loss improvement rate
        if len(losses) > 1:
            loss_diff = np.diff(losses)
            axes[1, 0].plot(loss_diff, 'g-', linewidth=2)
            axes[1, 0].set_title('Loss Improvement Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss Change')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 4. Training statistics
        stats_text = f"""Training Statistics:
        
Total Epochs: {len(losses)}
Initial Loss: {losses[0]:.6f}
Final Loss: {losses[-1]:.6f}
Best Loss: {losses.min():.6f}
Loss Reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%

Convergence:
{'✅ Converged' if losses[-1] < 1e-4 else '⚠️ May need more training'}
        """
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / "training_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  📊 Training analysis: {plot_path}")
        
        return fig
    
    def plot_prediction_analysis(self, results: Dict):
        """Create prediction analysis plots"""
        print("📊 Creating prediction analysis plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('GNN-ODE Prediction Analysis', fontsize=16)
        
        # 1 & 2. Trajectory plots for each person
        for idx, (person_name, result) in enumerate([(k, v) for k, v in results.items() if k != 'summary'][:2]):
            ax = axes[0, idx]
            
            times = result['times'].numpy()
            observed = result['observed_zones'].numpy()
            predicted = result['predicted_zones'].numpy()
            
            ax.plot(times, observed, 'o-', label='Observed', linewidth=2, markersize=4)
            ax.plot(times, predicted, 's--', label='Predicted', linewidth=2, markersize=4, alpha=0.7)
            
            ax.set_title(f'{person_name.title()} - Accuracy: {result["accuracy"]:.1%}')
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Zone')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.5, 7.5)
        
        # 3. Accuracy comparison
        ax = axes[0, 2]
        people = [k for k in results.keys() if k != 'summary']
        accuracies = [results[p]['accuracy'] for p in people]
        
        bars = ax.bar(people, accuracies, color=['skyblue', 'lightcoral'])
        ax.set_title('Accuracy by Person')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.1%}', ha='center', va='bottom')
        
        # 4. Zone distribution (observed vs predicted)
        ax = axes[1, 0]
        all_observed = torch.cat([results[p]['observed_zones'] for p in people])
        all_predicted = torch.cat([results[p]['predicted_zones'] for p in people])
        
        zones = range(8)
        obs_counts = [(all_observed == z).sum().item() for z in zones]
        pred_counts = [(all_predicted == z).sum().item() for z in zones]
        
        x = np.arange(len(zones))
        width = 0.35
        
        ax.bar(x - width/2, obs_counts, width, label='Observed', alpha=0.8)
        ax.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        
        ax.set_title('Zone Visit Distribution')
        ax.set_xlabel('Zone')
        ax.set_ylabel('Count')
        ax.set_xticks(x)
        ax.set_xticklabels(zones)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Physics violations analysis
        ax = axes[1, 1]
        violation_rate = results['summary']['violation_rate']
        compliance_rate = 1 - violation_rate
        
        labels = ['Physics Compliant', 'Physics Violations']
        sizes = [compliance_rate, violation_rate]
        colors = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Physics Compliance')
        
        # 6. Summary statistics
        ax = axes[1, 2]
        summary = results['summary']
        
        stats_text = f"""Model Performance Summary:
        
Overall Accuracy: {summary['overall_accuracy']:.1%}
Physics Violations: {summary['total_violations']}/{summary['total_transitions']} ({summary['violation_rate']:.1%})

Per Person:
"""
        
        for person in people:
            stats_text += f"  {person.title()}: {results[person]['accuracy']:.1%}\n"
        
        if summary['violation_rate'] == 0:
            status = "🎉 Perfect Physics Compliance!"
        elif summary['violation_rate'] < 0.1:
            status = "✅ Good Physics Compliance"
        elif summary['violation_rate'] < 0.3:
            status = "⚠️ Moderate Physics Violations"
        else:
            status = "❌ High Physics Violations"
        
        stats_text += f"\nStatus: {status}"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / "prediction_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  📊 Prediction analysis: {plot_path}")
        
        return fig
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting Complete GNN-ODE Analysis")
        print("=" * 50)
        
        # 1. Load model
        if not self.load_model():
            print("❌ Cannot proceed without a trained model!")
            return
        
        # 2. Evaluate model
        results = self.evaluate_model()
        
        # 3. Save predictions
        self.save_predictions_csv(results)
        
        # 4. Create visualizations
        self.plot_training_analysis()
        self.plot_prediction_analysis(results)
        
        # 5. Print final summary
        print("\n" + "=" * 50)
        print("🎯 ANALYSIS COMPLETE")
        print("=" * 50)
        
        summary = results['summary']
        print(f"📊 Overall Accuracy: {summary['overall_accuracy']:.1%}")
        print(f"🏗️  Physics Compliance: {100-summary['violation_rate']*100:.1f}%")
        print(f"📁 Results saved to: {self.results_dir}")
        print(f"🎨 Plots saved as PNG files")
        print(f"📄 Predictions saved as CSV files")
        
        return results

def main():
    """Main analysis function"""
    analyzer = GNNODEAnalyzer()
    results = analyzer.run_complete_analysis()
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = main() 
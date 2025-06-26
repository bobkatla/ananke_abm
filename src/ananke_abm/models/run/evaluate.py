"""
This script is dedicated to evaluating the best trained STG-NODE model.
It loads the best model checkpoint, runs evaluation, and generates outputs
like trajectory plots and prediction CSVs.
"""

import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Import the necessary classes from the training script
from ananke_abm.models.run.household_stg_node import (
    STGNodeHousehold,
    HouseholdDataProcessor,
    HouseholdConfig
)

# --- COPIED FROM household_stg_node.py ---

def evaluate_model(model, data, processor, config, adjacency_matrix):
    """Evaluate the trained model and return detailed results."""
    print("\n🔍 Evaluating Model Performance")
    print("=" * 50)
    
    model.eval()
    
    person_results = {}
    total_correct = 0
    total_predictions = 0
    total_violations = 0
    
    with torch.no_grad():
        trajectories_data = data['trajectories_data']
        times_data = data['times_data']
        
        for person_idx in range(data['num_people']):
            person_trajectory = trajectories_data[person_idx]
            person_times = times_data[person_idx]
            
            print(f"\n   Person {person_idx + 1} ({data['person_names'][person_idx]}):")
            print("   Time | True Zone | Pred Zone | Match | Physics OK")
            print("   -----|-----------|-----------|-------|----------")
            
            person_correct = 0
            person_preds = 0
            predicted_zones_list = []
            
            if len(person_trajectory) > 1:
                initial_zones = torch.tensor([person_trajectory[0].item() for _ in range(data['num_people'])])
                initial_time = person_times[0]
                eval_times = person_times
                
                raw_predictions = model(initial_zones, initial_time, eval_times)
                # The prediction from the new model might not need this apply_physics_constraints
                # For now, we assume it's part of the model's responsibility.
                # If apply_physics_constraints is a method on the model, this will work.
                try:
                    constrained_predictions = model.apply_physics_constraints(raw_predictions, initial_zones)
                except AttributeError:
                     # Fallback if the method doesn't exist on the new model
                    print("   (Note: `apply_physics_constraints` not found, using raw predictions)")
                    constrained_predictions = raw_predictions

                predicted_zones = torch.argmax(constrained_predictions[:, person_idx, :], dim=-1)
                
                for t in range(len(person_trajectory)):
                    true_zone = person_trajectory[t].item()
                    pred_zone = predicted_zones[t].item()
                    predicted_zones_list.append(pred_zone)
                    
                    true_zone_name = str(processor.id_to_zone.get(true_zone, 'N/A'))
                    pred_zone_name = str(processor.id_to_zone.get(pred_zone, 'N/A'))
                    
                    match = "✓" if true_zone == pred_zone else "✗"
                    if true_zone == pred_zone:
                        person_correct += 1
                    
                    if t > 0:
                        prev_pred_zone = predicted_zones[t-1].item()
                        is_compliant = (adjacency_matrix[prev_pred_zone, pred_zone] == 1 or prev_pred_zone == pred_zone)
                        physics_ok = "✓" if is_compliant else "✗"
                        if not is_compliant:
                            total_violations += 1
                    else:
                        physics_ok = "✓"
                    
                    person_preds += 1
                    time_val = person_times[t].item() if t < len(person_times) else t
                    print(f"   {time_val:4.1f} | {true_zone_name:9s} | {pred_zone_name:9s} | {match:5s} | {physics_ok:8s}")
            
            total_correct += person_correct
            total_predictions += person_preds
            person_results[person_idx] = {
                'accuracy': person_correct / person_preds if person_preds > 0 else 0,
                'predicted_traj': predicted_zones_list,
                'true_traj': person_trajectory.tolist(),
                'times': person_times.tolist()
            }

    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    violation_rate = total_violations / total_predictions if total_predictions > 0 else 0
    
    print(f"\n📈 Overall Zone Prediction Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.1f}%)")
    print(f"⚠️  Physics Violation Rate: {violation_rate:.4f} ({violation_rate*100:.1f}%) ({total_violations}/{total_predictions})")
    
    return {
        'overall_accuracy': overall_accuracy,
        'violation_rate': violation_rate,
        'person_results': person_results,
        'total_violations': total_violations,
        'total_predictions': total_predictions
    }

def visualize_results(eval_results, data, processor, config, plot_path=None):
    """Visualize the prediction results in a dashboard."""
    print("\n🎨 Visualizing Results Dashboard...")
    
    num_people = data['num_people']
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    # Plot trajectories for first 2 people
    for i in range(min(2, num_people)):
        ax = fig.add_subplot(gs[0, i])
        person_data = eval_results['person_results'][i]
        acc = person_data['accuracy']
        
        ax.plot(person_data['times'], person_data['true_traj'], 'o-', label='True', linewidth=2, markersize=5)
        ax.plot(person_data['times'], person_data['predicted_traj'], 's--', label='Predicted', linewidth=2, markersize=5)
        ax.set_title(f"{data['person_names'][i]}'s Trajectory - Accuracy: {acc:.1%}")
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Zone ID')
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_ylim(-0.5, config.num_zones - 0.5)

    # Plot accuracy bar chart
    ax_bar = fig.add_subplot(gs[1, 0])
    accuracies = [res['accuracy'] for res in eval_results['person_results'].values()]
    person_names = data['person_names']
    colors = plt.cm.viridis(np.linspace(0, 1, num_people))
    
    bars = ax_bar.bar(person_names, accuracies, color=colors)
    ax_bar.set_title('Accuracy by Person')
    ax_bar.set_ylabel('Accuracy')
    ax_bar.set_ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.1%}', ha='center', va='bottom')

    # Plot physics compliance pie chart
    ax_pie = fig.add_subplot(gs[1, 1])
    total_preds = eval_results['total_predictions']
    total_violations = eval_results['total_violations']
    compliant_preds = total_preds - total_violations
    
    sizes = [compliant_preds, total_violations]
    labels = [f'Compliant ({compliant_preds})', f'Violations ({total_violations})']
    colors = ['lightgreen', 'lightcoral']
    explode = (0, 0.1) if total_violations > 0 else (0, 0)

    ax_pie.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax_pie.axis('equal')
    ax_pie.set_title(f'Physics Compliance ({total_preds} total predictions)')

    fig.suptitle('GNN-ODE Evaluation Dashboard', fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if plot_path:
        plt.savefig(plot_path, dpi=300)
        print(f"📊 Dashboard saved to '{plot_path}'")
    plt.show()

# --- MAIN INFERENCE LOGIC ---

def main():
    """Main function to run model evaluation."""
    print("🚀 Starting Model Evaluation Script")
    print("=" * 60)
    
    try:
        # Define paths
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'saved_models')
        best_model_path = os.path.join(save_dir, 'household_stg_node_best.pth')
        
        if not os.path.exists(best_model_path):
            print(f"❌ Error: Best model not found at {best_model_path}")
            print("Please run the training script first.")
            return

        print(f"🔄 Loading best model from: {best_model_path}")
        
        # --- FIX: Load with weights_only=False as we trust the source ---
        best_model_data = torch.load(best_model_path, weights_only=False)
        
        # Re-create components needed for evaluation
        config = best_model_data['config']
        
        processor = HouseholdDataProcessor()
        # Restore the processor's state from the checkpoint
        processor.id_to_zone = best_model_data['processor_id_to_zone']
        data = processor.process_data()
        
        adjacency_matrix = data['adjacency_matrix']

        # Re-create the model and load its state
        model = STGNodeHousehold(config, data['num_zones'], data['zone_features'], data['person_features'], data['edge_index'])
        model.load_state_dict(best_model_data['model_state_dict'])
        
        # If the model has the set_adjacency_matrix method, call it.
        if hasattr(model, 'set_adjacency_matrix'):
            model.set_adjacency_matrix(adjacency_matrix)
        
        # Run evaluation
        eval_results = evaluate_model(model, data, processor, config, adjacency_matrix)
        
        # --- Save Outputs ---
        plot_dir = os.path.join(save_dir, 'plots')
        csv_path = os.path.join(save_dir, 'predictions_sample.csv')
        plot_path = os.path.join(plot_dir, 'evaluation_dashboard.png')

        # Save trajectory predictions to CSV
        records = []
        for i in range(min(2, data['num_people'])):
            person_res = eval_results['person_results'][i]
            for t_idx, time in enumerate(person_res['times']):
                records.append({
                    'person_id': i,
                    'person_name': data['person_names'][i],
                    'time': time,
                    'true_zone': processor.id_to_zone.get(person_res['true_traj'][t_idx], 'N/A'),
                    'predicted_zone': processor.id_to_zone.get(person_res['predicted_traj'][t_idx], 'N/A')
                })
        pd.DataFrame(records).to_csv(csv_path, index=False)
        print(f"🧾 Predictions for 2 persons saved to {csv_path}")

        # Visualize results
        visualize_results(eval_results, data, processor, config, plot_path=plot_path)
        
    except Exception as e:
        print(f"❌ Error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
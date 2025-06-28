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
    STG_CVAE,
    HouseholdDataProcessor,
    HouseholdConfig
)

# --- COPIED FROM household_stg_node.py ---

def evaluate_model(model, data, processor, config, adjacency_matrix, num_samples=5):
    """
    Evaluate the trained CVAE model.
    Generates multiple trajectory samples and calculates Best-of-N accuracy.
    """
    print(f"\n🔍 Evaluating Model Performance (generating {num_samples} samples per trajectory)")
    print("=" * 70)
    
    model.eval()
    
    person_results = {}
    total_best_of_n_correct = 0
    total_predictions = 0
    
    with torch.no_grad():
        trajectories_data = data['trajectories_data']
        times_data = data['times_data']
        
        for person_idx in range(data['num_people']):
            person_trajectory = trajectories_data[person_idx]
            person_times = times_data[person_idx]
            
            print(f"\n   Person {person_idx + 1} ({data['person_names'][person_idx]}):")
            
            if len(person_trajectory) < 2:
                continue

            initial_zones = torch.tensor([person_trajectory[0].item() for _ in range(data['num_people'])])
            initial_time = person_times[0]
            eval_times = person_times

            # Store all generated trajectories for this person
            all_predicted_trajectories = []

            for s in range(num_samples):
                # For evaluation, we sample a batch of z vectors from a standard normal distribution,
                # one for each person in the household.
                z_batch = torch.randn(data['num_people'], config.latent_dim)
                
                # Decode to get a single trajectory prediction
                raw_predictions = model.decoder(initial_zones, initial_time, eval_times, z_batch)
                
                # Apply hard physics constraints to the person we are currently evaluating
                constrained_logits = raw_predictions.clone()
                predicted_zones = torch.zeros(len(eval_times), dtype=torch.long)
                for t in range(len(eval_times)):
                    prev_zone = initial_zones[person_idx].item() if t == 0 else predicted_zones[t-1].item()
                    valid_mask = adjacency_matrix[prev_zone].clone()
                    valid_mask[prev_zone] = 1.0
                    invalid_mask = (valid_mask == 0)
                    constrained_logits[t, person_idx, invalid_mask] = -1e9
                    predicted_zones[t] = torch.argmax(constrained_logits[t, person_idx, :])
                
                all_predicted_trajectories.append(predicted_zones.tolist())

            # Calculate Best-of-N accuracy
            person_correct = 0
            for t in range(len(person_trajectory)):
                true_zone = person_trajectory[t].item()
                # Check if any of the N samples got it right
                if any(sample[t] == true_zone for sample in all_predicted_trajectories):
                    person_correct += 1
            
            total_best_of_n_correct += person_correct
            total_predictions += len(person_trajectory)
            
            person_results[person_idx] = {
                'accuracy': person_correct / len(person_trajectory) if len(person_trajectory) > 0 else 0,
                'predicted_trajs': all_predicted_trajectories, # Note the plural
                'true_traj': person_trajectory.tolist(),
                'times': person_times.tolist()
            }
            print(f"   Best-of-{num_samples} Accuracy: {person_results[person_idx]['accuracy']:.1%}")

    overall_accuracy = total_best_of_n_correct / total_predictions if total_predictions > 0 else 0
    
    print(f"\n📈 Overall Best-of-{num_samples} Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.1f}%)")
    
    return {
        'overall_accuracy': overall_accuracy,
        'person_results': person_results,
    }

def visualize_results(eval_results, data, processor, config, plot_path=None):
    """Visualize the prediction results in a dashboard."""
    print("\n🎨 Visualizing Results Dashboard...")
    
    num_people = data['num_people']
    num_samples = len(list(eval_results['person_results'].values())[0]['predicted_trajs'])
    fig = plt.figure(figsize=(16, 6 * num_people))
    
    for i in range(num_people):
        ax = fig.add_subplot(num_people, 1, i + 1)
        person_data = eval_results['person_results'][i]
        acc = person_data['accuracy']
        
        # Plot multiple predicted trajectories with transparency
        for j, pred_traj in enumerate(person_data['predicted_trajs']):
            ax.plot(person_data['times'], pred_traj, '--', label=f'Sample {j+1}' if i==0 else "", alpha=0.6)
            
        # Plot the true trajectory on top
        ax.plot(person_data['times'], person_data['true_traj'], 'o-', label='True', linewidth=3, markersize=6, color='black', zorder=10)
        
        ax.set_title(f"{data['person_names'][i]}'s Trajectory - Best-of-{num_samples} Accuracy: {acc:.1%}")
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Zone ID')
        ax.grid(True, alpha=0.4)
        ax.set_ylim(-0.5, config.num_zones - 0.5)

    # Create a single legend for the first plot
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    
    fig.suptitle('CVAE Generative Evaluation Dashboard', fontsize=20, y=1.02)
    fig.tight_layout()
    
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
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
        model = STG_CVAE(
            config, 
            data['num_zones'], 
            data['zone_features'], 
            data['person_features'], 
            data['edge_index_phys'], 
            data['edge_index_sem']
        )
        model.load_state_dict(best_model_data['model_state_dict'])
        
        # The set_adjacency_matrix method has been removed.
        # The adjacency matrix is passed directly to evaluation functions.
        
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
            # Save the first predicted sample along with the true trajectory
            first_pred_traj = person_res['predicted_trajs'][0]
            for t_idx, time in enumerate(person_res['times']):
                records.append({
                    'person_id': i,
                    'person_name': data['person_names'][i],
                    'time': time,
                    'true_zone': processor.id_to_zone.get(person_res['true_traj'][t_idx], 'N/A'),
                    'predicted_zone_sample_1': processor.id_to_zone.get(first_pred_traj[t_idx], 'N/A')
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
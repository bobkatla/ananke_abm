"""
This script is dedicated to evaluating the best trained CDE-STGNN model.
It loads the best model checkpoint, runs a generative evaluation, and generates
outputs like trajectory plots and prediction CSVs.
"""

import torch
import torchcde
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

from ananke_abm.models.run.household_stg_node import (
    CDE_STGNN,
    HouseholdDataProcessor,
    HouseholdConfig,
    get_device,
)
from ananke_abm.utils.helpers import pad_sequences


def evaluate_model(model, data, processor, config, adjacency_matrix):
    """
    Evaluate the trained CDE-STGNN model using a time-aware, autoregressive,
    generative approach.
    """
    print(f"\n🔍 Evaluating Model Performance (Time-Aware Autoregressive Generation)")
    print("=" * 70)
    
    model.eval()
    device = next(model.parameters()).device
    
    person_results = {}
    
    with torch.no_grad():
        true_trajectories = data['trajectories_y'].to(device)
        times = data['times'].to(device)
        num_people = data['num_people']
        person_features_raw = data['person_features_raw'].to(device)
        max_len = true_trajectories.shape[1]

        # --- Dynamically generate features, similar to the training loop ---
        phys_zone_embeds, sem_zone_embeds = model.get_gnn_embeds()

        person_embeds = model.person_feature_embedder(person_features_raw)
        home_zone_ids = true_trajectories[:, 0]
        home_phys_embeds = phys_zone_embeds[home_zone_ids]
        home_sem_embeds = sem_zone_embeds[home_zone_ids]
        static_features = torch.cat([person_embeds, home_phys_embeds, home_sem_embeds], dim=1)

        # Get initial hidden state h0 from the first true observation and static features
        initial_gnn_embed = torch.cat([
            phys_zone_embeds[true_trajectories[:, 0]],
            sem_zone_embeds[true_trajectories[:, 0]]
        ], dim=1)
        h0 = torch.tanh(model.initial_state_mapper(torch.cat([static_features, initial_gnn_embed], dim=1)))
        
        h_t = h0
        generated_trajectory = [true_trajectories[:, 0].unsqueeze(1)]

        print("   Generating trajectories step-by-step...")
        for t in range(max_len - 1):
            # --- Construct the augmented control path for a single step ---
            last_obs_zones = generated_trajectory[-1].squeeze()
            
            # 1. Get GNN embeddings for the last observed zone
            phys_path_t = phys_zone_embeds[last_obs_zones]
            sem_path_t = sem_zone_embeds[last_obs_zones]
            gnn_path_t = torch.cat([phys_path_t, sem_path_t], dim=1)

            # 2. Get the current time and static features
            time_t = times[t].view(1, 1).expand(num_people, -1)
            static_path_t = static_features

            # 3. Combine into a single vector for the path segment
            # Shape for one time step: (batch, feature_dim)
            X_vec_t = torch.cat([time_t, static_path_t, gnn_path_t], dim=1).unsqueeze(1)
            
            # 4. Create a path segment of length 2 (from t to t+1) to solve the CDE
            X_step = torch.cat([X_vec_t, X_vec_t], dim=1)
            t_step = times[t:t+2]
            
            # 5. Interpolate and solve
            X_path_step = torchcde.CubicSpline(torchcde.hermite_cubic_coefficients_with_backward_differences(X_step))
            h_t_plus_1 = torchcde.cdeint(X=X_path_step, func=model.cde_func, z0=h_t, t=t_step)
            
            h_t = h_t_plus_1[:, 1, :]
            
            pred_logits = model.predictor(h_t)
            pred_zone = torch.argmax(pred_logits, dim=1, keepdim=True)
            
            generated_trajectory.append(pred_zone)

        generated_trajectory = torch.cat(generated_trajectory, dim=1)
        
        correct_preds = (generated_trajectory == true_trajectories).float().sum()
        total_preds = num_people * max_len
        overall_accuracy = correct_preds / total_preds

        print(f"\n📈 Overall Generative Accuracy: {overall_accuracy.item():.4f} ({overall_accuracy.item()*100:.1f}%)")
        
        for i in range(num_people):
            person_results[i] = {
                'accuracy': (generated_trajectory[i] == true_trajectories[i]).float().mean().item(),
                'predicted_traj': generated_trajectory[i].tolist(),
                'true_traj': true_trajectories[i].tolist(),
                'times': times.tolist()
            }

    return {
        'overall_accuracy': overall_accuracy.item(),
        'person_results': person_results,
    }


def visualize_results(eval_results, data, processor, config, plot_path=None):
    """Visualize the prediction results in a dashboard."""
    print("\n🎨 Visualizing Results Dashboard...")
    
    num_people = data['num_people']
    fig = plt.figure(figsize=(16, 6 * num_people))
    
    for i in range(num_people):
        ax = fig.add_subplot(num_people, 1, i + 1)
        person_data = eval_results['person_results'][i]
        acc = person_data['accuracy']
        
        ax.plot(person_data['times'], person_data['predicted_traj'], 'x--', label='Predicted', alpha=0.8)
        ax.plot(person_data['times'], person_data['true_traj'], 'o-', label='True', linewidth=3, markersize=6, color='black', zorder=10)
        
        ax.set_title(f"{data['person_names'][i]}'s Generated Trajectory - Accuracy: {acc:.1%}")
        ax.set_xlabel('Time')
        ax.set_ylabel('Zone ID')
        ax.grid(True, alpha=0.4)
        ax.set_ylim(-0.5, config.num_zones - 0.5)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    
    fig.suptitle('CDE-STGNN Generative Evaluation', fontsize=20, y=1.02)
    fig.tight_layout()
    
    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Dashboard saved to '{plot_path}'")
    plt.show()

def main():
    """Main function to run model evaluation."""
    print("🚀 Starting Model Evaluation Script (Time-Aware CDE-STGNN)")
    print("=" * 60)
    
    try:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'saved_models')
        best_model_path = os.path.join(save_dir, 'cde_stgnn_best.pth')
        
        if not os.path.exists(best_model_path):
            print(f"❌ Error: Best model not found at {best_model_path}")
            print("Please run the training script first.")
            return

        device = get_device()
        print(f"🔄 Loading best model from: {best_model_path} onto device: {device}")
        
        config = HouseholdConfig()
        processor = HouseholdDataProcessor()
        
        # We process data first, as it now defines the model's dimensions
        data = processor.process_data()
        config.num_zones = data['num_zones']
        
        model = CDE_STGNN(
            config,
            data['num_zones'],
            data['person_features_raw'].shape[1],
            data['zone_features'],
            data['edge_index_phys'],
            data['edge_index_sem']
        ).to(device)
        
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        
        # Add torch.compile for potential speedup
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("✅ Evaluation model compiled successfully with torch.compile!")
        except Exception as e:
            print(f"⚠️ Could not compile evaluation model with torch.compile: {e}. Continuing without it.")
        
        eval_results = evaluate_model(model, data, processor, config, data['adjacency_matrix'])
        
        plot_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, 'cde_predictions.csv')
        plot_path = os.path.join(plot_dir, 'cde_evaluation_dashboard.png')

        records = []
        for i in range(data['num_people']):
            res = eval_results['person_results'][i]
            for t_idx, time in enumerate(res['times']):
                records.append({
                    'person_id': i,
                    'person_name': data['person_names'][i],
                    'time': time,
                    'true_zone': res['true_traj'][t_idx],
                    'predicted_zone': res['predicted_traj'][t_idx]
                })
        pd.DataFrame(records).to_csv(csv_path, index=False)
        print(f"🧾 Predictions saved to {csv_path}")

        visualize_results(eval_results, data, processor, config, plot_path=plot_path)
        
    except Exception as e:
        print(f"❌ Error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
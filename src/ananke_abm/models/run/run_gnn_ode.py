#!/usr/bin/env python3
"""
Runner script for the Batched Multi-Agent GNN-ODE Model.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch

# Make the project root directory available
# This allows us to import from ananke_abm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

try:
    from ananke_abm.data_generator.load_data import load_mobility_data
    from ananke_abm.data_generator.mock_locations import create_mock_zone_graph
    from ananke_abm.models.gnn_embed.graph_utils import prepare_household_batch
    from ananke_abm.models.gnn_embed.gnn_ode import PhysicsGNNODE, GNNODETrainer, ModelTracker
except ImportError as e:
    print("--- IMPORT ERROR ---")
    print(f"Failed to import a module. This is often a path issue.")
    print(f"Current sys.path: {sys.path}")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def plot_results(tracker: ModelTracker, save_dir: Path):
    """Plots training statistics and saves them."""
    print("📊 Plotting training results...")
    
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Loss
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(tracker.training_losses, color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Instantiate a second y-axis for Accuracy
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.plot(tracker.accuracies, color=color, linestyle='--', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    if tracker.learning_rates:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        color = 'tab:green'
        ax3.set_ylabel('Learning Rate', color=color)
        ax3.plot(tracker.learning_rates, color=color, linestyle=':', label='Learning Rate')
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.set_yscale('log')

    fig.tight_layout()
    plt.title('GNN-ODE Training Statistics')
    
    # Collect all labels for a single legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = (ax3.get_legend_handles_labels() if tracker.learning_rates else ([], []))
    ax2.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='best')

    save_path = save_dir / "gnn_ode_training_stats.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Training plot saved to {save_path}")

def main():
    """Main execution function."""
    SAVE_DIR = Path("saved_models/gnn_ode_run")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Load and Prepare Data ---
    # This function returns trajectories in a dict and people/zones as DataFrames
    trajectories_dict, people_df, zones_df = load_mobility_data()
    
    # Convert the trajectories dict to a single DataFrame for batch preparation
    traj_list = []
    for person_name, data in trajectories_dict.items():
        for t, z in zip(data['times'], data['zones']):
            traj_list.append({
                'person_id': data['person_id'],
                'time': t,
                'zone_id': z
            })
    trajectories_df = pd.DataFrame(traj_list)

    # Define a single household for Sarah (ID 1) and Marcus (ID 2)
    people_df['household_id'] = 0
    people_df.loc[people_df['person_id'] == 1, 'household_id'] = 101
    people_df.loc[people_df['person_id'] == 2, 'household_id'] = 101
    
    batch, true_trajectories, zone_id_to_idx, common_times = prepare_household_batch(trajectories_df, people_df)
    
    person_id_to_idx = {pid: i for i, pid in enumerate(sorted(people_df['person_id'].unique()))}

    # Create the static location graph structure for the model
    # The zones_df doesn't have edge info, so we get it from the source
    zone_graph, _ = create_mock_zone_graph()
    location_edge_index = torch.tensor(list(zone_graph.edges), dtype=torch.long).t().contiguous()
    # Adjust for 1-based indexing if necessary (it's 1-based in the mock data)
    if location_edge_index.min() == 1:
        location_edge_index = location_edge_index - 1

    # --- 2. Initialize Model and Trainer ---
    model = PhysicsGNNODE(
        num_people=batch.num_nodes,
        num_zones=len(zone_id_to_idx),
        location_edge_index=location_edge_index,
        embedding_dim=32,
        person_feature_dim=batch.x.shape[1]
    )
    
    trainer = GNNODETrainer(model, lr=0.01, save_dir=SAVE_DIR)

    # --- 3. Train the Model ---
    print("\n--- 🚀 Starting Model Training ---")
    trainer.train(batch, true_trajectories, person_id_to_idx, common_times, num_epochs=300)
    print("--- ✅ Model Training Finished ---\n")

    # --- 4. Evaluate and Save Predictions ---
    print("--- 🔬 Evaluating Final Model ---")
    trainer.load_best_model() 
    results = trainer.evaluate(batch, true_trajectories, person_id_to_idx, common_times)
    print(f"Final Model Accuracy: {results['accuracy']:.2f}%")

    all_preds = []
    for person_id, data in results['predictions'].items():
        person_name = "Sarah" if person_id == 1 else "Marcus"
        df = pd.DataFrame({
            'time': data['times'],
            'predicted_zone_id': [list(zone_id_to_idx.keys())[i] for i in data['pred_zones']],
            'true_zone_id': [list(zone_id_to_idx.keys())[i] for i in data['true_zones']]
        })
        df['person_id'] = person_id
        df['person_name'] = person_name
        all_preds.append(df)
    
    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_save_path = SAVE_DIR / "final_predictions.csv"
    pred_df.to_csv(pred_save_path, index=False)
    print(f"✅ Predictions saved to {pred_save_path}")

    # --- 5. Save Artifacts ---
    trainer.tracker.save_training_data()
    plot_results(trainer.tracker, SAVE_DIR)
    
    print("\n--- ✨ Run Complete ---")
    print(f"All artifacts saved in: {SAVE_DIR.resolve()}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
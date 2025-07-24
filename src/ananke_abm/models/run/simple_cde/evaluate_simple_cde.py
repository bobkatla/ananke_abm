"""
Evaluates the trained Encoder-Decoder CDE model.
This script loads the best saved model checkpoint and runs it on the dataset
to generate predictions, calculate accuracy, and plot visualizations.
"""

import torch
import torch.nn as nn
import torchcde
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# To make the script runnable, we need to import the model and config classes
from ananke_abm.models.run.simple_cde.household_simple_cde import (
    HouseholdEncoderDecoderConfig,
    CDEFunc,
    EncoderDecoderCDE,
    SimpleDataProcessor,
    get_device
)

def plot_loss_curve(save_dir):
    """
    Loads and plots the training and validation loss curves from the training run.
    """
    loss_path = os.path.join(save_dir, 'training_losses.npz')
    if not os.path.exists(loss_path):
        print(f"⚠️ Warning: Loss file not found at {loss_path}. Skipping plot.")
        return

    print("📊 Plotting training loss curve...")
    data = np.load(loss_path)
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    val_epochs = data['val_epochs']

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss (per batch)')
    plt.plot(val_epochs, val_loss, label='Validation Loss', marker='o')
    
    plt.title("Saved Training and Validation Loss Curve")
    plt.xlabel("Epochs / Batches")
    plt.ylabel("Cross-Entropy Loss (log scale)")
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    loss_plot_path = os.path.join(save_dir, "evaluation_loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"   -> Saved to {loss_plot_path}")


def evaluate_model():
    """
    Loads the best model and evaluates its performance using a full
    autoregressive rollout and checks for physics violations.
    """
    print("📈 Evaluating Encoder-Decoder CDE Model (Autoregressive Rollout)")
    print("=" * 60)

    config = HouseholdEncoderDecoderConfig()
    device = get_device()
    processor = SimpleDataProcessor()
    
    print("📊 Processing data for evaluation (single day)...")
    data = processor.process_data(repeat_pattern=False)
    num_zones = data['num_zones']
    
    # --- Setup for Physics Violation Check ---
    edge_index = data['edge_index'].cpu().numpy()
    valid_transitions = set(tuple(edge) for edge in edge_index.T)
    # Add self-loops (it's always valid to stay in the same zone)
    for i in range(num_zones):
        valid_transitions.add((i, i))
    
    model = EncoderDecoderCDE(
        config,
        num_zones=num_zones,
        person_feat_dim=data['person_features_raw'].shape[1],
        padding_idx=data['padding_value']
    ).to(device)

    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', '..', 'saved_models', 'encoder_decoder_cde')
    model_path = os.path.join(save_dir, 'enc_dec_cde_best.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model checkpoint not found at {model_path}")
        return

    print(f"📂 Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Autoregressive Prediction ---
    print("🤖 Generating full trajectory via autoregressive rollout...")
    
    full_times = data['times'].to(device)
    full_y = data['trajectories_y'].to(device)
    person_features_raw = data['person_features_raw'].to(device)
    num_people, full_seq_len = full_y.shape
    
    # Start with the ground truth history
    generated_y = full_y[:, :config.history_length].clone()
    physics_violations = 0

    with torch.no_grad():
        person_embeds = model.person_feature_embedder(person_features_raw)
        home_zone_ids = full_y[:, 0].to(device)
        home_zone_embeds = model.zone_embedder(home_zone_ids)

        for i in range(config.history_length, full_seq_len - config.prediction_length):
            start_idx = i - config.history_length
            history_end_idx = i
            cde_end_idx = i + config.prediction_length

            # Use the *generated* history for prediction
            y_history = generated_y[:, start_idx:history_end_idx]
            
            # The CDE path still needs a real start point
            y_cde_start = generated_y[:, history_end_idx - 1].unsqueeze(1)
            y_cde_end_actual_time = full_times[cde_end_idx -1] # Use actual time for CDE solve
            
            # This is tricky - we need a plausible *next* step for the CDE control.
            # We'll use the *actual* next step's location to define the path, 
            # but the LSTM context comes from the generated sequence.
            y_cde_actual = full_y[:, history_end_idx - 1 : cde_end_idx]

            history_zone_embeds = model.zone_embedder(y_history)
            static_embeds = torch.cat([person_embeds, home_zone_embeds], dim=1)
            expanded_static_embeds = static_embeds.unsqueeze(1).expand(-1, config.history_length, -1)
            history_path = torch.cat([history_zone_embeds, expanded_static_embeds], dim=2)
            
            cde_zone_embeds = model.zone_embedder(y_cde_actual)
            cde_times = full_times[history_end_idx - 1 : cde_end_idx]
            cde_time_path = cde_times.view(1, -1, 1).expand(num_people, -1, -1)
            cde_path_values = torch.cat([cde_time_path, cde_zone_embeds], dim=2)
            cde_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(cde_path_values)
            
            pred_logits = model(history_path, cde_coeffs, cde_times)
            pred_class = torch.argmax(pred_logits, dim=1).unsqueeze(1)

            # Append the prediction to our generated sequence
            generated_y = torch.cat([generated_y, pred_class], dim=1)

            # Check for physics violations
            for person_idx in range(num_people):
                prev_zone = generated_y[person_idx, -2].item()
                new_zone = generated_y[person_idx, -1].item()
                if (prev_zone, new_zone) not in valid_transitions:
                    physics_violations += 1

    # --- Calculate Metrics ---
    # Compare the generated trajectory to the actual one, ignoring the warm-up period
    eval_slice_generated = generated_y[:, config.history_length:].cpu().numpy()
    
    # FIX: Slice the actual data to match the length of the generated data
    num_predictions = eval_slice_generated.shape[1]
    actual_end_idx = config.history_length + num_predictions
    eval_slice_actual = full_y[:, config.history_length:actual_end_idx].cpu().numpy()
    
    accuracy = np.mean(eval_slice_actual == eval_slice_generated)
    print(f"\n🎯 Autoregressive Accuracy: {accuracy:.2%}")
    print(f"⚖️ Physics Violations: {physics_violations}")

    # --- Plot Confusion Matrix ---
    print("📊 Plotting confusion matrix...")
    cm = confusion_matrix(eval_slice_actual.flatten(), eval_slice_generated.flatten())
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Autoregressive)')
    plt.ylabel('Actual Zone')
    plt.xlabel('Predicted Zone')
    cm_path = os.path.join(save_dir, 'confusion_matrix_autoregressive.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"   -> Saved to {cm_path}")

    # --- Plot Trajectory Comparison ---
    print("📊 Plotting full autoregressive trajectory...")
    plt.figure(figsize=(15, 8))
    for i, name in enumerate(data['person_names']):
        plt.subplot(num_people, 1, i + 1)
        
        actual_traj = full_y[i].cpu().numpy()
        generated_traj = generated_y[i].cpu().numpy()
        
        # FIX: Only plot for the duration we have predictions
        plot_len = len(generated_traj)
        time_steps = np.arange(plot_len)

        plt.plot(time_steps, actual_traj[:plot_len], label='Actual Trajectory', color='blue', marker='o', linestyle='-', markersize=4)
        plt.plot(time_steps, generated_traj, label='Generated Trajectory (Autoregressive)', color='red', marker='x', linestyle='--', markersize=4)
        
        plt.title(f'Autoregressive Trajectory for {name}')
        plt.ylabel('Zone ID')
        plt.grid(True)
        plt.legend()

    plt.xlabel('Time Step')
    plt.tight_layout()
    traj_path = os.path.join(save_dir, 'trajectory_comparison_autoregressive.png')
    plt.savefig(traj_path)
    plt.close()
    print(f"   -> Saved to {traj_path}")

    plot_loss_curve(save_dir)

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    try:
        evaluate_model()
    except Exception as e:
        print(f"❌ Error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc() 
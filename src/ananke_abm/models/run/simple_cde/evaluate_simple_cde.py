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
import pandas as pd

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

    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', '..', 'saved_models', 'encoder_decoder_cde22')
    model_path = os.path.join(save_dir, 'enc_dec_cde_best.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model checkpoint not found at {model_path}")
        return

    print(f"📂 Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Autoregressive Prediction with Rejection Sampling ---
    print("🤖 Generating trajectories via rejection sampling...")

    # Get data and move to device
    full_times = data['times'].to(device)
    time_features = data['time_features'].to(device)
    full_y = data['trajectories_y'].to(device)
    person_features_raw = data['person_features_raw'].to(device)
    people_edge_index = data['people_edge_index'].to(device)
    num_people, full_seq_len = full_y.shape
    
    max_retries = 10

    # Pre-compute embeddings for all people once
    with torch.no_grad():
        person_embeds_all = model.person_feature_embedder(person_features_raw)
        social_context = model.social_gnn(person_embeds_all, people_edge_index)
        person_embeds_all = person_embeds_all + social_context
        home_zone_ids_all = full_y[:, 0].to(device)
        home_zone_embeds_all = model.zone_embedder(home_zone_ids_all)

    # Store results
    accepted_trajectories = []
    all_violation_masks = []
    total_physics_violations = 0

    # Main loop over each person
    for person_idx in range(num_people):
        print(f"   -> Generating for person {person_idx + 1}/{num_people}...")
        
        best_trajectory_for_person = None
        min_violations_for_person = float('inf')
        best_violations_mask_for_person = []

        # Rejection sampling loop
        for retry in range(max_retries):
            # Start with the ground truth history for this person
            generated_y_person = full_y[person_idx, :config.history_length].clone().unsqueeze(0)
            current_violations = 0
            violations_mask = [False] * full_seq_len # Init mask for the full length
            
            # Autoregressive generation loop for the full trajectory
            with torch.no_grad():
                for i in range(config.history_length, full_seq_len):
                    history_end_idx = i

                    y_history = generated_y_person[:, -config.history_length:]
                    person_embeds = person_embeds_all[person_idx].unsqueeze(0)
                    home_zone_embeds = home_zone_embeds_all[person_idx].unsqueeze(0)
                    
                    # --- Create a self-driven CDE path without looking at ground truth ---
                    last_generated_zone = generated_y_person[:, -1:]
                    y_cde_path_plausible = last_generated_zone.repeat(1, config.prediction_length)
                    cde_zone_embeds = model.zone_embedder(y_cde_path_plausible)

                    # Handle time features for the final step to prevent index error
                    if i < full_seq_len - 1:
                        cde_time_feats = time_features[i-1 : i-1 + config.prediction_length]
                    else: # On the last step, just repeat the last known time feature
                        cde_time_feats = time_features[i-1:i].repeat(1, 2).view(config.prediction_length, -1)

                    cde_times = torch.linspace(0, 1, config.prediction_length).to(device)

                    history_zone_embeds = model.zone_embedder(y_history)
                    static_embeds = torch.cat([person_embeds, home_zone_embeds], dim=1)
                    expanded_static_embeds = static_embeds.unsqueeze(1).expand(-1, config.history_length, -1)
                    history_path = torch.cat([history_zone_embeds, expanded_static_embeds], dim=2)
                    
                    cde_time_path = cde_time_feats.unsqueeze(0)
                    
                    cde_path_values = torch.cat([cde_time_path, cde_zone_embeds], dim=2)
                    cde_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(cde_path_values)
                    
                    # --- STOCHASTIC PREDICTION ---
                    pred_logits = model(history_path, cde_coeffs, cde_times)
                    probabilities = torch.nn.functional.softmax(pred_logits, dim=-1)
                    pred_class = torch.multinomial(probabilities, num_samples=1)

                    generated_y_person = torch.cat([generated_y_person, pred_class], dim=1)

                    # Check for physics violations for this step
                    prev_zone = generated_y_person[0, -2].item()
                    new_zone = generated_y_person[0, -1].item()
                    if (prev_zone, new_zone) not in valid_transitions:
                        current_violations += 1
                        violations_mask[i] = True # Mark violation at the current step
            
            # After generating a full trajectory, check for acceptance
            if current_violations == 0:
                min_violations_for_person = 0
                best_trajectory_for_person = generated_y_person.squeeze(0)
                best_violations_mask_for_person = violations_mask
                break
            
            if current_violations < min_violations_for_person:
                min_violations_for_person = current_violations
                best_trajectory_for_person = generated_y_person.squeeze(0)
                best_violations_mask_for_person = violations_mask
        
        if min_violations_for_person > 0:
            print(f"   ⚠️  Warning: Could not find a 0-violation trajectory for person {person_idx + 1}. Best had {min_violations_for_person} violations.")
        
        accepted_trajectories.append(best_trajectory_for_person)
        all_violation_masks.append(best_violations_mask_for_person)
        total_physics_violations += min_violations_for_person

    # Combine accepted trajectories into a single tensor for evaluation
    generated_y = torch.stack(accepted_trajectories)

    # --- Calculate Metrics ---
    # Compare the generated trajectory to the actual one, ignoring the warm-up period
    eval_slice_generated = generated_y[:, config.history_length:].cpu().numpy()
    
    # FIX: Slice the actual data to match the length of the generated data
    num_predictions = eval_slice_generated.shape[1]
    actual_end_idx = config.history_length + num_predictions
    eval_slice_actual = full_y[:, config.history_length:actual_end_idx].cpu().numpy()
    
    accuracy = np.mean(eval_slice_actual == eval_slice_generated)
    print(f"\n🎯 Autoregressive Accuracy: {accuracy:.2%}")
    print(f"⚖️ Total Physics Violations: {total_physics_violations}")

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

    # --- Save detailed trajectory data to CSV for debugging ---
    save_trajectories_to_csv(full_y, generated_y, all_violation_masks, data['person_names'], save_dir)

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


def save_trajectories_to_csv(actual_y, generated_y, violation_masks, person_names, save_dir):
    """
    Saves a detailed comparison of actual vs. generated trajectories to a CSV file.
    
    Args:
        actual_y (Tensor): The ground truth trajectories.
        generated_y (Tensor): The generated trajectories.
        violation_masks (list of list of bool): For each person, a list indicating if a violation occurred at that step.
        person_names (list of str): The names of the people.
        save_dir (str): The directory to save the file in.
    """
    print("📋 Saving detailed trajectory comparison to CSV...")
    
    num_people, seq_len = actual_y.shape
    gen_seq_len = generated_y.shape[1]
    
    records = []
    for p_idx in range(num_people):
        for t_idx in range(seq_len):
            actual_zone_val = actual_y[p_idx, t_idx].item()
            is_violation = violation_masks[p_idx][t_idx] if t_idx < len(violation_masks[p_idx]) else False
            
            # Check if the generated trajectory is long enough for this timestep
            if t_idx < gen_seq_len:
                generated_zone_val = generated_y[p_idx, t_idx].item()
                is_correct_val = actual_zone_val == generated_zone_val
            else:
                generated_zone_val = 'N/A' # Use a placeholder for missing steps
                is_correct_val = False

            records.append({
                "Person": person_names[p_idx],
                "Time_Step": t_idx,
                "Actual_Zone": actual_zone_val,
                "Generated_Zone": generated_zone_val,
                "Is_Violation": is_violation,
                "Is_Correct": is_correct_val
            })
            
    df = pd.DataFrame(records)
    csv_path = os.path.join(save_dir, 'trajectory_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"   -> Saved to {csv_path}")


if __name__ == "__main__":
    try:
        evaluate_model()
    except Exception as e:
        print(f"❌ Error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc() 
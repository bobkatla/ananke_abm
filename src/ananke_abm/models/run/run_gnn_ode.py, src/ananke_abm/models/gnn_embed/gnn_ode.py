# Define a single household for Sarah (ID 1) and Marcus (ID 2)
people_df['household_id'] = 0
people_df.loc[people_df['person_id'] == 1, 'household_id'] = 101
people_df.loc[people_df['person_id'] == 2, 'household_id'] = 101

# The new function returns the common time vector as well
batch, true_trajectories, zone_id_to_idx, common_times = prepare_household_batch(trajectories_df, people_df)

person_id_to_idx = {pid: i for i, pid in enumerate(sorted(people_df['person_id'].unique()))}

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

def train_step(self, batch: Batch, true_trajectories: Dict, person_id_to_idx: Dict, times: torch.Tensor) -> float:
    """Single training step on a batch of households."""
    self.model.train()
    self.optimizer.zero_grad()
    
    pred_trajectory = self.model(batch, times)
    loss = self.model.compute_loss(pred_trajectory, true_trajectories, person_id_to_idx)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    self.optimizer.step()
    
    return loss.item()

def train(self, batch: Batch, true_trajectories: Dict, person_id_to_idx: Dict, 
          times: torch.Tensor, num_epochs: int = 200, verbose: bool = True):
    """Train the model"""
    
    for epoch in range(num_epochs):
        loss = self.train_step(batch, true_trajectories, person_id_to_idx, times)
        
        # Evaluate on the training data itself
        self.model.eval()
        with torch.no_grad():
            pred_trajectory = self.model(batch, times)
            
            # Calculate accuracy
            distances = torch.cdist(pred_trajectory, self.model.location_embeddings.weight)
            pred_zones = torch.argmin(distances, dim=2) # Shape: [T, N]

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}, LR: {lr:.6f}, Accuracy: {accuracy:.2f}%")

def evaluate(self, batch: Batch, true_trajectories: Dict, person_id_to_idx: Dict, times: torch.Tensor) -> Dict:
    """Evaluates the model on a given batch."""
    self.model.eval()
    results = {"predictions": {}, "accuracy": 0.0}

    with torch.no_grad():
        pred_trajectory_embed = self.model(batch, times)

        distances = torch.cdist(pred_trajectory_embed, self.model.location_embeddings.weight)
        pred_zones = torch.argmin(distances, dim=2) # Shape: [T, N]

        # Calculate accuracy
        correct = (pred_zones == true_trajectories['zone_id']).sum().item()
        total = true_trajectories['zone_id'].size(0) * true_trajectories['zone_id'].size(1)
        accuracy = (correct / total) * 100

        results["accuracy"] = accuracy
        results["predictions"] = {
            "pred_zones": pred_zones,
            "true_zones": true_trajectories['zone_id']
        }

    return results 
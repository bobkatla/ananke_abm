"""
Script for evaluating a trained Generative Latent ODE model (OLD VERSION - before mode choice).
This script evaluates models trained with the old architecture that only predicts location and purpose.
"""
import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ananke_abm.models.latent_ode.config import GenerativeODEConfig
from ananke_abm.models.latent_ode.data_process.data import DataProcessor

# Define the OLD model architecture components
class ResidualBlock(nn.Module):
    """A residual block with two linear layers and a skip connection."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(x + self.net(x))

class OldODEFunc(nn.Module):
    """
    OLD ODE function with the original state vector: [h, y_loc, y_purp]
    """
    def __init__(self, state_dim, hidden_dim, num_residual_blocks, 
                 zone_embed_dim=8, purpose_embed_dim=4, enable_attention=True, attention_strength=0.1):
        super().__init__()
        # OLD: state = [h, loc_embed, purp_embed]
        self.actual_hidden_dim = state_dim - zone_embed_dim - purpose_embed_dim
        self.ode_hidden_dim = hidden_dim
        self.zone_embed_dim = zone_embed_dim
        self.purpose_embed_dim = purpose_embed_dim
        self.enable_attention = enable_attention
        self.attention_strength = attention_strength
        
        # Core network (original structure)
        self.net = nn.Sequential(
            nn.Linear(state_dim + 2, self.ode_hidden_dim), # +2 for sine/cosine time features
            nn.ReLU(),
            *[ResidualBlock(self.ode_hidden_dim) for _ in range(num_residual_blocks)],
            nn.Linear(self.ode_hidden_dim, state_dim)
        )
        
        # Attention mechanisms (original)
        if self.enable_attention:
            min_embed_dim = min(zone_embed_dim, purpose_embed_dim)
            self.loc_purp_attention = nn.MultiheadAttention(
                embed_dim=min_embed_dim, num_heads=2, batch_first=True
            )
            
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=self.actual_hidden_dim, num_heads=2, batch_first=True
            )
            
            # Projection layers for dimension alignment
            self.loc_proj = nn.Linear(zone_embed_dim, min_embed_dim) if zone_embed_dim != min_embed_dim else nn.Identity()
            self.purp_proj = nn.Linear(purpose_embed_dim, min_embed_dim) if purpose_embed_dim != min_embed_dim else nn.Identity()
            self.loc_back_proj = nn.Linear(min_embed_dim, zone_embed_dim) if zone_embed_dim != min_embed_dim else nn.Identity()
        
        self.static_features = None
        self.home_zone_embedding = None

    def forward(self, t, state):
        # state is the OLD combined vector [h, y_loc, y_purp]
        batch_size = state.shape[0]
        
        if t.dim() == 0:
            t = t.expand(1)
        t_vec = torch.cat([torch.sin(t * 2 * torch.pi / 24), torch.cos(t * 2 * torch.pi / 24)]).expand(batch_size, -1)
        
        core_dynamics = self.net(torch.cat([state, t_vec], dim=-1))
        
        if self.enable_attention:
            attention_delta = self._compute_attention_delta(state, t_vec)
            return core_dynamics + self.attention_strength * attention_delta
        else:
            return core_dynamics
    
    def _compute_attention_delta(self, state, t_vec):
        # Split state into OLD components
        h = state[:, :self.actual_hidden_dim]
        loc_embed = state[:, self.actual_hidden_dim:self.actual_hidden_dim + self.zone_embed_dim]
        purp_embed = state[:, self.actual_hidden_dim + self.zone_embed_dim:]
        
        delta_h = torch.zeros_like(h)
        delta_loc = torch.zeros_like(loc_embed)
        delta_purp = torch.zeros_like(purp_embed)
        
        # Original location-purpose attention
        loc_purp_delta = self._location_purpose_attention(loc_embed, purp_embed)
        delta_loc += loc_purp_delta
        
        # Temporal attention
        temporal_delta = self._temporal_state_attention(h, t_vec)
        delta_h += temporal_delta
        
        return torch.cat([delta_h, delta_loc, delta_purp], dim=-1)
    
    def _location_purpose_attention(self, loc_embed, purp_embed):
        loc_proj = self.loc_proj(loc_embed).unsqueeze(1)
        purp_proj = self.purp_proj(purp_embed).unsqueeze(1)
        
        enhanced_loc_proj, _ = self.loc_purp_attention(
            query=purp_proj, key=loc_proj, value=loc_proj
        )
        
        delta_proj = enhanced_loc_proj.squeeze(1) - self.loc_proj(loc_embed)
        delta_original = self.loc_back_proj(delta_proj)
        
        return delta_original
    
    def _temporal_state_attention(self, hidden_state, time_features):
        hidden_expanded = hidden_state.unsqueeze(1)
        time_expanded = time_features.unsqueeze(1)
        
        time_padded = torch.cat([
            time_expanded, 
            torch.zeros(time_expanded.shape[0], 1, self.actual_hidden_dim - 2).to(time_expanded.device)
        ], dim=-1)
        
        enhanced_hidden, _ = self.temporal_attention(
            query=time_padded, key=hidden_expanded, value=hidden_expanded
        )
        
        return enhanced_hidden.squeeze(1) - hidden_state

class OldGenerativeODE(nn.Module):
    """
    OLD Generative ODE model with original state vector [h, y_loc, y_purp]
    """
    def __init__(self, person_feat_dim, num_zone_features, config):
        super().__init__()
        self.config = config
        
        self.zone_feature_encoder = nn.Linear(num_zone_features, config.zone_embed_dim)
        self.purpose_embedder = nn.Embedding(len(config.purpose_groups), config.purpose_embed_dim)

        encoder_input_dim = person_feat_dim + config.zone_embed_dim * 2
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, config.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.encoder_hidden_dim, config.hidden_dim * 2),
        )
        
        # OLD state dimensions: [hidden_state, location_embed, purpose_embed]
        self.state_dim = config.hidden_dim + config.zone_embed_dim + config.purpose_embed_dim
        
        self.ode_func = OldODEFunc(
            state_dim=self.state_dim,
            hidden_dim=config.ode_hidden_dim,
            num_residual_blocks=config.num_residual_blocks,
            zone_embed_dim=config.zone_embed_dim,
            purpose_embed_dim=config.purpose_embed_dim,
            enable_attention=config.enable_attention,
            attention_strength=config.attention_strength
        )
        
        self.decoder_loc = nn.Linear(config.hidden_dim, config.zone_embed_dim)
        self.decoder_purp_logits = nn.Linear(config.hidden_dim, len(config.purpose_groups))

    def forward(self, person_features, home_zone_features, work_zone_features, 
                start_purp_id, times, all_zone_features, adjacency_matrix):
        
        candidate_zone_embeds = self.zone_feature_encoder(all_zone_features)
        home_embed = self.zone_feature_encoder(home_zone_features)
        work_embed = self.zone_feature_encoder(work_zone_features)
        
        encoder_input = torch.cat([person_features, home_embed, work_embed], dim=-1)
        h0_params = self.encoder(encoder_input)
        mu, log_var = h0_params.chunk(2, dim=-1)
        h0 = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

        # OLD: Initial state with purpose embedding
        s0_loc = home_embed
        s0_purp = self.purpose_embedder(start_purp_id)
        s0 = torch.cat([h0, s0_loc, s0_purp], dim=-1)

        pred_s = odeint(self.ode_func, s0, times, method=self.config.ode_method).permute(1, 0, 2)
        
        pred_h = pred_s[..., :self.config.hidden_dim]
        pred_y_loc_embed_path = pred_s[..., self.config.hidden_dim : self.config.hidden_dim + self.config.zone_embed_dim]
        
        target_loc_embeds = self.decoder_loc(pred_h)
        pred_y_loc_logits = torch.einsum('bsd,zd->bsz', target_loc_embeds, candidate_zone_embeds)
        pred_y_purp_logits = self.decoder_purp_logits(pred_h)
        
        return pred_y_loc_logits, pred_y_loc_embed_path, pred_y_purp_logits, mu, log_var

def evaluate_old():
    """Loads a trained model (old architecture) and generates evaluation plots."""
    config = GenerativeODEConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = DataProcessor(device, config)
    print(f"🔬 Using device: {device}")
    print(f"📊 Evaluating OLD model (before mode choice integration)")

    # --- Model Initialization with OLD architecture ---
    init_data = processor.get_data(person_id=1)
    model = OldGenerativeODE(  # Use OLD model class
        person_feat_dim=init_data["person_features"].shape[-1],
        num_zone_features=init_data["all_zone_features"].shape[-1],
        config=config,
    ).to(device)
    
    # --- Load Trained Model ---
    folder_path = Path("saved_models/generative_ode_batched")
    model_path = folder_path / "latent_ode_best_model_batched.pth"
    print(f"📈 Evaluating old model from '{model_path}'...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Model loaded successfully with matching architecture")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}. Please run train.py first.")
        return
        
    # --- Plot Training Loss (Old Format) ---
    training_stats_path = folder_path / "latent_ode_training_stats_batched.npz"
    try:
        stats = np.load(training_stats_path)
        
        plt.figure(figsize=(14, 8))
        
        # Old loss components (before mode loss)
        old_loss_keys = {
            'total_loss': 'Total Loss',
            'classification_loss': 'Location Classification',
            'embedding_loss': 'Location Embedding',
            'distance_loss': 'Physical Distance',
            'purpose_loss': 'Purpose Classification',
            'kl_loss': 'KL Divergence'
        }
        
        for key, label in old_loss_keys.items():
            if key in stats:
                plt.plot(stats[key], label=label, alpha=0.9)

        plt.title("Training Loss Components (OLD MODEL - Before Mode Choice)")
        plt.xlabel("Iteration")
        plt.ylabel("Average Loss (Log Scale)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.yscale('log')
        plt.tight_layout()
        loss_plot_path = folder_path / "old_training_loss_curves_batched.png"
        plt.savefig(loss_plot_path)
        print(f"   📉 Old training loss plots saved to '{loss_plot_path}'")
        plt.close()
            
    except FileNotFoundError:
        print(f"WARNING: Training stats file not found at {training_stats_path}. Skipping loss plot.")

    model.eval()

    person_ids = [1, 2]

    for person_id in person_ids:
        with torch.no_grad():
            data = processor.get_data(person_id=person_id)
            person_name = data['person_name']
            print(f"   -> Generating trajectory for {person_name}...")

            person_features = data["person_features"].unsqueeze(0)
            home_zone_features = data["home_zone_features"].unsqueeze(0)
            work_zone_features = data["work_zone_features"].unsqueeze(0)
            start_purpose_id = torch.tensor([data["start_purpose_id"]], device=device)
            all_zone_features = data["all_zone_features"]
            adjacency_matrix = data["adjacency_matrix"]

            plot_times = torch.linspace(0, 24, 100).to(device)
            
            # OLD model outputs (exact original format)
            pred_y_logits, _, pred_purpose_logits, _, _ = model(
                person_features, home_zone_features, work_zone_features, 
                start_purpose_id, plot_times, all_zone_features, adjacency_matrix
            )
            
            pred_y = torch.argmax(pred_y_logits.squeeze(0), dim=1)
            pred_purpose = torch.argmax(pred_purpose_logits.squeeze(0), dim=1)

        # --- Original 2-Panel Visualization (before mode choice) ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Location plot
        ax1.plot(data["times"].cpu().numpy(), data["trajectory_y"].cpu().numpy(), 'o', label='Ground Truth Location', markersize=8)
        ax1.plot(plot_times.cpu().numpy(), pred_y.cpu().numpy(), '-', label='Generated Location')
        ax1.set_ylabel("Zone ID")
        ax1.set_title(f"Generated vs. Ground Truth for {person_name} (OLD MODEL - Before Mode Choice)")
        ax1.set_yticks(np.arange(data["num_zones"]))
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.legend()
        
        # Purpose plot  
        ax2.plot(data["times"].cpu().numpy(), data["target_purpose_ids"].cpu().numpy(), 'o', label='Ground Truth Purpose', markersize=8)
        ax2.plot(plot_times.cpu().numpy(), pred_purpose.cpu().numpy(), '-', label='Generated Purpose')
        ax2.set_xlabel("Time (hours)")
        ax2.set_ylabel("Purpose ID")
        ax2.set_yticks(np.arange(len(config.purpose_groups)))
        ax2.set_yticklabels(config.purpose_groups, rotation=30, ha='right')
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.legend()
        
        plt.tight_layout()
        
        save_path = folder_path / f"old_generative_ode_trajectory_{person_name.replace(' ', '_')}_batched.png"
        plt.savefig(save_path)
        print(f"   📄 Old model plot saved to '{save_path}'")
        plt.close()
        
        # --- Original Statistics (no mode analysis) ---
        ground_truth_purposes = data["target_purpose_ids"].cpu().numpy()
        print(f"   📊 Purpose analysis for {person_name}:")
        purpose_names = config.purpose_groups
        print(f"      Ground truth purpose distribution: {dict(zip(purpose_names, [np.sum(ground_truth_purposes == i) for i in range(len(purpose_names))]))}")
        
        # Calculate purpose transition statistics
        purpose_transitions = []
        for i in range(len(ground_truth_purposes) - 1):
            if ground_truth_purposes[i] != ground_truth_purposes[i+1]:
                purpose_transitions.append((purpose_names[ground_truth_purposes[i]], purpose_names[ground_truth_purposes[i+1]]))
        
        print(f"      Purpose transitions: {len(purpose_transitions)} transitions")
        if purpose_transitions:
            print(f"      Most common transitions: {purpose_transitions[:3]}")

    print("✅ Old model evaluation complete.")

if __name__ == "__main__":
    evaluate_old() 
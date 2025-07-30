"""
Model architecture for the Generative Latent ODE.
"""
import torch
import torch.nn as nn
from torchdiffeq import odeint

from .config import GenerativeODEConfig

class ResidualBlock(nn.Module):
    """A residual block with two linear layers and a skip connection."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(x + self.net(x))

class MovementAwareDecoder(nn.Module):
    """Structured decoder that enforces stay ↔ transit mutual exclusion"""
    def __init__(self, hidden_dim, num_purposes, num_modes):
        super().__init__()
        self.num_purposes = num_purposes
        self.num_modes = num_modes
        
        # Core movement intention (binary choice: staying vs moving)
        self.movement_net = nn.Linear(hidden_dim, 1)
        
        # Purpose selection conditioned on movement
        self.staying_purpose_net = nn.Linear(hidden_dim, num_purposes - 1)  # Exclude transit
        
        # Mode selection conditioned on movement  
        self.moving_mode_net = nn.Linear(hidden_dim, num_modes - 1)  # Exclude stay
        
    def forward(self, hidden_state):
        # 1. Fundamental movement intention
        movement_logit = self.movement_net(hidden_state)
        movement_prob = torch.sigmoid(movement_logit)
        
        # 2. Purpose logits with structural constraint
        staying_purpose_logits = self.staying_purpose_net(hidden_state)
        # Combine: staying purposes weighted by (1-movement), transit weighted by movement
        purpose_logits = torch.cat([
            staying_purpose_logits * (1 - movement_prob),  # Home, Work, Subsistence, Leisure, Social
            movement_logit  # Transit purpose (high when moving)
        ], dim=-1)
        
        # 3. Mode logits with structural constraint
        moving_mode_logits = self.moving_mode_net(hidden_state)
        stay_logit = -movement_logit  # Inverse of movement intention
        # Combine: stay weighted by (1-movement), moving modes weighted by movement
        mode_logits = torch.cat([
            stay_logit,  # Stay mode (high when not moving)
            moving_mode_logits * movement_prob  # Walk, Car, Public_Transit
        ], dim=-1)
        
        return purpose_logits, mode_logits, movement_prob

class ODEFunc(nn.Module):
    """
    The dynamics function for the conditional ODE. It computes the derivative of
    the combined state [h, y_loc, movement_intention] at time t.
    Enhanced with optional attention mechanisms.
    """
    def __init__(self, state_dim, hidden_dim, num_residual_blocks, 
                 zone_embed_dim=8, enable_attention=True, attention_strength=0.1):
        super().__init__()
        # Calculate the actual hidden dim from state structure: state = [h, loc_embed, movement_intention]
        self.actual_hidden_dim = state_dim - zone_embed_dim - 1  # -1 for movement_intention
        self.ode_hidden_dim = hidden_dim  # For the network layers
        self.zone_embed_dim = zone_embed_dim
        self.enable_attention = enable_attention
        self.attention_strength = attention_strength
        
        # Core network (adapted for new state structure)
        self.net = nn.Sequential(
            nn.Linear(state_dim + 2, self.ode_hidden_dim), # +2 for sine/cosine time features
            nn.ReLU(),
            *[ResidualBlock(self.ode_hidden_dim) for _ in range(num_residual_blocks)],
            nn.Linear(self.ode_hidden_dim, state_dim)
        )
        
        # Attention mechanisms (adapted for movement-aware state)
        if self.enable_attention:
            # Location-Movement Cross-Attention
            self.loc_movement_attention = nn.MultiheadAttention(
                embed_dim=self.zone_embed_dim, num_heads=2, batch_first=True
            )
            
            # Temporal-State Cross-Attention  
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=self.actual_hidden_dim, num_heads=2, batch_first=True
            )
        
        self.static_features = None
        self.home_zone_embedding = None

    def forward(self, t, state):
        # state is the combined vector [h, y_loc, movement_intention]
        batch_size = state.shape[0]
        
        # Ensure t is a tensor and then expand for sine/cosine features
        if t.dim() == 0:
            t = t.expand(1)
        t_vec = torch.cat([torch.sin(t * 2 * torch.pi / 24), torch.cos(t * 2 * torch.pi / 24)]).expand(batch_size, -1)
        
        # Core dynamics
        core_dynamics = self.net(torch.cat([state, t_vec], dim=-1))
        
        # Apply attention enhancements if enabled
        if self.enable_attention:
            attention_delta = self._compute_attention_delta(state, t_vec)
            return core_dynamics + self.attention_strength * attention_delta
        else:
            return core_dynamics
    
    def _compute_attention_delta(self, state, t_vec):
        """Compute attention-based modifications to the dynamics."""
        # Split state into components
        h = state[:, :self.actual_hidden_dim]
        loc_embed = state[:, self.actual_hidden_dim:self.actual_hidden_dim + self.zone_embed_dim]
        movement_intention = state[:, -1:]
        
        # Initialize delta with zeros
        delta_h = torch.zeros_like(h)
        delta_loc = torch.zeros_like(loc_embed)
        delta_movement = torch.zeros_like(movement_intention)
        
        # 1. Location-Movement Cross-Attention
        loc_movement_delta = self._location_movement_attention(loc_embed, movement_intention)
        delta_loc += loc_movement_delta
        
        # 2. Temporal-State Attention  
        temporal_delta = self._temporal_state_attention(h, t_vec)
        delta_h += temporal_delta
        
        return torch.cat([delta_h, delta_loc, delta_movement], dim=-1)
    
    def _location_movement_attention(self, loc_embed, movement_intention):
        """Cross-attention between location and movement intention."""
        # Expand movement intention to match location embedding dimension
        movement_expanded = movement_intention.expand(-1, self.zone_embed_dim).unsqueeze(1)
        loc_expanded = loc_embed.unsqueeze(1)
        
        # Movement as query, location as key/value
        enhanced_loc, _ = self.loc_movement_attention(
            query=movement_expanded, key=loc_expanded, value=loc_expanded
        )
        
        # Calculate delta
        delta = enhanced_loc.squeeze(1) - loc_embed
        
        return delta
    
    def _temporal_state_attention(self, hidden_state, time_features):
        """Temporal attention to modulate hidden state dynamics."""
        # Expand dimensions for attention
        hidden_expanded = hidden_state.unsqueeze(1)  # [batch, 1, hidden_dim]
        time_expanded = time_features.unsqueeze(1)   # [batch, 1, 2]
        
        # Pad time features to match hidden dimension
        time_padded = torch.cat([
            time_expanded, 
            torch.zeros(time_expanded.shape[0], 1, self.actual_hidden_dim - 2).to(time_expanded.device)
        ], dim=-1)
        
        # Time as query, hidden state as key/value
        enhanced_hidden, _ = self.temporal_attention(
            query=time_padded, key=hidden_expanded, value=hidden_expanded
        )
        
        # Return the delta
        return enhanced_hidden.squeeze(1) - hidden_state

class GenerativeODE(nn.Module):
    """
    A conditional, autoregressive Latent ODE model with movement-aware mode choice.
    The ODE state contains the evolving location embedding and movement intention.
    """
    def __init__(self, person_feat_dim, num_zone_features, config):
        super().__init__()
        self.config = config
        
        self.zone_feature_encoder = nn.Linear(num_zone_features, config.zone_embed_dim)

        # The encoder produces the initial hidden state h(0)
        # It now takes zone *characteristics* instead of embeddings
        encoder_input_dim = person_feat_dim + config.zone_embed_dim * 2
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, config.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.encoder_hidden_dim, config.hidden_dim * 2), # mu and log_var for h(0)
        )
        
        # Define the dimensions of the combined ODE state
        # State: [hidden_state, location_embed, movement_intention]
        self.state_dim = config.hidden_dim + config.zone_embed_dim + 1
        
        self.ode_func = ODEFunc(
            state_dim=self.state_dim,
            hidden_dim=config.ode_hidden_dim,
            num_residual_blocks=config.num_residual_blocks,
            zone_embed_dim=config.zone_embed_dim,
            enable_attention=config.enable_attention,
            attention_strength=config.attention_strength
        )
        
        # Decoders now operate on the hidden state component h(t)
        # The location decoder predicts a "target embedding", not logits over all zones
        self.decoder_loc = nn.Linear(config.hidden_dim, config.zone_embed_dim)
        
        # NEW: Structured decoder for purpose and mode with constraints
        self.structured_decoder = MovementAwareDecoder(
            config.hidden_dim, len(config.purpose_groups), config.num_modes
        )

    def forward(self, person_features, home_zone_features, work_zone_features, 
                start_purp_id, times, all_zone_features, adjacency_matrix):
        
        # Create candidate embeddings for all zones in the geography
        candidate_zone_embeds = self.zone_feature_encoder(all_zone_features)

        # Encode home and work zones from their features
        home_embed = self.zone_feature_encoder(home_zone_features)
        work_embed = self.zone_feature_encoder(work_zone_features)
        
        # 1. ENCODER: Get initial hidden state h(0) from static features
        encoder_input = torch.cat([person_features, home_embed, work_embed], dim=-1)
        h0_params = self.encoder(encoder_input)
        mu, log_var = h0_params.chunk(2, dim=-1)
        h0 = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

        # 2. INITIAL STATE: Construct the full initial state s(0)
        # State: [hidden_state, location_embed, movement_intention]
        s0_loc = home_embed # Start at home location
        s0_movement = torch.zeros_like(h0[:, :1])  # Start stationary (movement_intention = 0)
        s0 = torch.cat([h0, s0_loc, s0_movement], dim=-1)

        # 3. SOLVER: Evolve the state s(t) over the entire time series
        pred_s = odeint(self.ode_func, s0, times, method=self.config.ode_method).permute(1, 0, 2)
        
        # 4. DECODE: Split the solved state back into its components
        pred_h = pred_s[..., :self.config.hidden_dim]
        pred_y_loc_embed_path = pred_s[..., self.config.hidden_dim:self.config.hidden_dim + self.config.zone_embed_dim]
        # movement_intention_path = pred_s[..., -1:]  # Available if needed for analysis
        
        # 5. PREDICT: Use the hidden state to predict a "target embedding"
        target_loc_embeds = self.decoder_loc(pred_h)

        # 6. MATCH: Find the most similar candidate zone for each predicted target embedding
        # This computes the dot product similarity and gives us our new logits
        pred_y_loc_logits = torch.einsum('bsd,zd->bsz', target_loc_embeds, candidate_zone_embeds)

        # 7. STRUCTURED PREDICTION: Purpose and mode with constraints
        pred_y_purp_logits, pred_y_mode_logits, movement_probs = self.structured_decoder(pred_h)
        
        return pred_y_loc_logits, pred_y_loc_embed_path, pred_y_purp_logits, pred_y_mode_logits, mu, log_var 
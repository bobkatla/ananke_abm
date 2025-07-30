"""
Demo script showing how to use SDE (Stochastic Differential Equations) 
with the Generative Latent ODE model for sharper transitions.
"""

from ananke_abm.models.latent_ode.config import GenerativeODEConfig
from ananke_abm.models.latent_ode.train.train import train

def demo_sde_training():
    """Example of training with SDE enabled for sharp transitions."""
    
    # Create config with SDE enabled
    config = GenerativeODEConfig(
        # Enable SDE for stochastic dynamics
        enable_sde=True,
        sde_noise_strength=0.08,  # Moderate noise for sharp transitions
        
        # You can adjust other parameters
        enable_attention=True,
        attention_strength=0.1,
        
        # Training parameters
        learning_rate=1e-3,
        num_iterations=3000,
        
        # Loss weights
        loss_weight_mode=1.2,  # Emphasize mode learning
    )
    
    print("🎲 Starting SDE training for sharp transition learning...")
    print(f"   Noise strength: {config.sde_noise_strength}")
    print(f"   Expected benefits:")
    print(f"   - Sharp mode switches (Stay ↔ Walk/Car)")
    print(f"   - Realistic decision timing")
    print(f"   - Reduced oversmoothing")
    print()
    
    # Train the model (same interface as before)
    train()

def demo_ode_vs_sde_comparison():
    """Compare ODE vs SDE behavior."""
    
    configs = {
        "ODE": GenerativeODEConfig(enable_sde=False),
        "SDE": GenerativeODEConfig(enable_sde=True, sde_noise_strength=0.05)
    }
    
    for mode, config in configs.items():
        print(f"\n🔄 Training with {mode} mode:")
        print(f"   SDE enabled: {config.enable_sde}")
        if config.enable_sde:
            print(f"   Noise strength: {config.sde_noise_strength}")
        
        # Here you would run training and compare results
        # train() would use the current config
        
if __name__ == "__main__":
    # Run SDE demo
    demo_sde_training()
    
    # Uncomment to compare modes
    # demo_ode_vs_sde_comparison() 
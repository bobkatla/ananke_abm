#!/usr/bin/env python3
"""
Training script for single-person zone dynamics model.
This is a simple test case to validate the core ODE approach.
"""

import sys
import os
# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import torch
from ananke_abm.data_generator.mock_1p import Person, create_mock_zone_graph, create_sarah_daily_pattern, create_training_data
from ananke_abm.models.run.simple_for1 import SimpleZoneODE, ZoneTrajectoryPredictor, train_zone_model, predict_and_visualize

def main():
    """Main training and evaluation pipeline"""
    
    print("=== Ananke ABM: Single-Person Zone Dynamics ===\n")
    
    # Create mock data
    print("📊 Creating mock data...")
    sarah = Person()
    zone_graph, zone_data = create_mock_zone_graph()
    sarah_schedule = create_sarah_daily_pattern()
    
    print(f"Person: {sarah.name}")
    print(f"Zones: {len(zone_graph.nodes())} zones, {len(zone_graph.edges())} connections")
    print(f"Schedule: {len(sarah_schedule)} time points")
    
    # Convert to training format
    print("\n🔄 Converting to training format...")
    training_data = create_training_data(sarah, sarah_schedule, zone_graph)
    
    print(f"Person attributes shape: {training_data['person_attrs'].shape}")
    print(f"Time observations: {len(training_data['times'])}")
    print(f"Zone features shape: {training_data['zone_features'].shape}")
    print(f"Edge features shape: {training_data['edge_features'].shape}")
    
    # Create model
    print("\n🧠 Creating neural ODE model...")
    ode_func = SimpleZoneODE()
    model = ZoneTrajectoryPredictor(ode_func)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("\n🚀 Training model...")
    losses = train_zone_model(model, training_data, epochs=500)
    
    # Make predictions
    print("\n📈 Making predictions...")
    predicted_zones, zone_probs = predict_and_visualize(model, training_data, sarah_schedule)
    
    # Final analysis
    actual_zones = training_data["zone_observations"]
    final_accuracy = torch.mean((predicted_zones == actual_zones).float()).item()
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Final accuracy: {final_accuracy:.3f} ({final_accuracy*100:.1f}%)")
    print(f"Training epochs: {len(losses)}")
    
    if len(losses) > 0:
        print(f"Final loss: {losses[-1]:.4f}")
    else:
        print("Training failed - no epochs completed")
    
    if final_accuracy > 0.95:
        print("✅ SUCCESS: Model achieved high accuracy!")
    elif final_accuracy > 0.8:
        print("⚠️  PARTIAL SUCCESS: Model learned main patterns")
    else:
        print("❌ FAILURE: Model did not learn the pattern well")
    
    return model, training_data, losses

if __name__ == "__main__":
    model, training_data, losses = main() 
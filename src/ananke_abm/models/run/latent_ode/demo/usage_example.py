"""
Simple usage example for the batched inference system.
Shows how to use the Generative Latent ODE model for practical mobility prediction.
"""

# Example 1: Quick prediction for a few people
from ananke_abm.models.run.latent_ode.inference import quick_inference

def example_quick_prediction():
    """Simple example of predicting trajectories for multiple people."""
    
    # Predict trajectories for 5 people over 24 hours
    person_ids = [1, 2, 1, 2, 1]  # Mix of different person types
    
    predictions = quick_inference(
        person_ids=person_ids,
        batch_size=5,
        time_resolution=24  # Hourly predictions
    )
    
    print(f"Predicted trajectories for {len(person_ids)} people")
    print(f"Locations shape: {predictions['locations'].shape}")  # [5 people, 24 hours]
    print(f"Modes shape: {predictions['modes'].shape}")          # [5 people, 24 hours]
    
    return predictions

# Example 2: Large-scale city simulation
from ananke_abm.models.run.latent_ode.inference import BatchedInferenceEngine

def example_city_simulation():
    """Example of simulating mobility for a large population."""
    
    # Initialize inference engine
    model_path = "saved_models/mode_generative_ode_batched/latent_ode_best_model_batched.pth"
    engine = BatchedInferenceEngine(model_path)
    
    # Simulate 1000 people
    population_size = 1000
    person_ids = [1 + (i % 2) for i in range(population_size)]
    
    # Generate 24-hour trajectories
    predictions = engine.predict_trajectories(
        person_ids=person_ids,
        time_resolution=24,
        batch_size=100  # Process 100 people at a time
    )
    
    print(f"City simulation complete for {population_size} people")
    return predictions

# Example 3: Custom time windows
def example_rush_hour_analysis():
    """Analyze mobility patterns during rush hour."""
    
    model_path = "saved_models/mode_generative_ode_batched/latent_ode_best_model_batched.pth"
    engine = BatchedInferenceEngine(model_path)
    
    # Focus on morning rush hour (7-9 AM)
    import torch
    rush_hour_times = torch.linspace(7.0, 9.0, 13).to(engine.device)  # Every 10 minutes
    
    # Analyze 200 commuters
    person_ids = [1, 2] * 100  # Alternate between person types
    
    predictions = engine.batch_inference(
        person_ids=person_ids,
        times=rush_hour_times,
        batch_size=50
    )
    
    # Analyze mode choice during rush hour
    mode_logits = predictions['mode_logits']
    pred_modes = torch.argmax(mode_logits, dim=-1).cpu().numpy()
    
    mode_names = ["Stay", "Walk", "Car", "Public_Transit"]
    for mode_id, mode_name in enumerate(mode_names):
        usage_rate = (pred_modes == mode_id).mean() * 100
        print(f"Rush hour {mode_name} usage: {usage_rate:.1f}%")
    
    return predictions

if __name__ == "__main__":
    print("🎯 Batched Inference Usage Examples")
    print("=" * 50)
    
    try:
        # Run examples
        print("\n1. Quick prediction example:")
        example_quick_prediction()
        
        print("\n2. City simulation example:")
        example_city_simulation()
        
        print("\n3. Rush hour analysis example:")
        example_rush_hour_analysis()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have a trained model available!") 
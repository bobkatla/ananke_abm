"""
Demo script showcasing the batched inference capabilities of the Generative Latent ODE model.
This demonstrates how to efficiently process thousands of people simultaneously.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ananke_abm.models.run.latent_ode.inference import BatchedInferenceEngine, quick_inference

def demo_quick_inference():
    """Demonstrate the quick inference function for immediate use."""
    print("🚀 Demo 1: Quick Inference Function")
    print("=" * 50)
    
    # Simple one-liner inference
    person_ids = [1, 2, 1, 2, 1]  # Example: process 5 people (cycling through available IDs)
    
    try:
        results = quick_inference(
            person_ids=person_ids,
            batch_size=32,
            time_resolution=50  # Lower resolution for speed
        )
        
        print(f"✅ Processed {len(person_ids)} people successfully!")
        print(f"   Output shapes:")
        print(f"   - Times: {results['times'].shape}")
        print(f"   - Locations: {results['locations'].shape}")
        print(f"   - Purposes: {results['purposes'].shape}")
        print(f"   - Modes: {results['modes'].shape}")
        print(f"   - Person names: {len(results['person_names'])}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure you have a trained model at the default path!")
        return None

def demo_scalability_test():
    """Demonstrate scalability across different population sizes."""
    print("\n🏁 Demo 2: Scalability Benchmarking")
    print("=" * 50)
    
    # Initialize inference engine
    model_path = "saved_models/mode_generative_ode_batched/latent_ode_best_model_batched.pth"
    
    try:
        engine = BatchedInferenceEngine(model_path)
        
        # Test different scales
        test_scales = [1, 5, 10, 25, 50, 100]
        engine.benchmark_performance(
            num_people_list=test_scales,
            batch_size=32,
            time_resolution=100
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure you have a trained model!")

def demo_large_population_simulation():
    """Demonstrate large population simulation capabilities."""
    print("\n🌆 Demo 3: Large Population Simulation")
    print("=" * 50)
    
    model_path = "saved_models/mode_generative_ode_batched/latent_ode_best_model_batched.pth"
    
    try:
        engine = BatchedInferenceEngine(model_path)
        
        # Simulate a larger population
        population_size = 200
        person_ids = [1 + (i % 2) for i in range(population_size)]  # Cycle through available person types
        
        print(f"🏙️  Simulating {population_size} people across 24 hours...")
        
        # Generate trajectories
        predictions = engine.predict_trajectories(
            person_ids=person_ids,
            time_resolution=24,  # Hourly resolution
            batch_size=50
        )
        
        # Analyze population-level patterns
        analyze_population_patterns(predictions, population_size)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure you have a trained model!")

def analyze_population_patterns(predictions, population_size):
    """Analyze and visualize population-level mobility patterns."""
    times = predictions['times']
    locations = predictions['locations']
    purposes = predictions['purposes']
    modes = predictions['modes']
    
    print(f"✅ Generated trajectories for {population_size} people")
    
    # Population-level analysis
    print("\n📊 Population-Level Analysis:")
    
    # 1. Mode choice distribution over time
    mode_names = ["Stay", "Walk", "Car", "Public_Transit"]
    mode_counts_by_time = np.zeros((len(times), len(mode_names)))
    
    for t_idx in range(len(times)):
        for mode_id in range(len(mode_names)):
            mode_counts_by_time[t_idx, mode_id] = np.sum(modes[:, t_idx] == mode_id)
    
    # 2. Activity patterns
    purpose_names = ["Home", "Work/Education", "Subsistence", "Leisure & Recreation", "Social", "Travel/Transit"]
    purpose_counts_by_time = np.zeros((len(times), len(purpose_names)))
    
    for t_idx in range(len(times)):
        for purpose_id in range(len(purpose_names)):
            purpose_counts_by_time[t_idx, purpose_id] = np.sum(purposes[:, t_idx] == purpose_id)
    
    # Print key insights
    peak_travel_hour = times[np.argmax(mode_counts_by_time[:, 1:].sum(axis=1))]  # Non-stay modes
    peak_home_hour = times[np.argmax(purpose_counts_by_time[:, 0])]  # Home activities
    
    print(f"   Peak travel time: {peak_travel_hour:.1f} hours")
    print(f"   Peak home time: {peak_home_hour:.1f} hours")
    print(f"   Total person-hours simulated: {population_size * 24}")
    
    # Create population-level visualization
    create_population_visualization(times, mode_counts_by_time, purpose_counts_by_time, 
                                   mode_names, purpose_names, population_size)

def create_population_visualization(times, mode_counts, purpose_counts, mode_names, purpose_names, population_size):
    """Create population-level mobility pattern visualization."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Mode choice patterns
    mode_percentages = mode_counts / population_size * 100
    
    # Stacked area plot for modes
    ax1.stackplot(times, mode_percentages.T, labels=mode_names, alpha=0.8)
    ax1.set_ylabel("Population Percentage (%)")
    ax1.set_title(f"Population Mode Choice Patterns ({population_size} people)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Purpose patterns
    purpose_percentages = purpose_counts / population_size * 100
    
    # Stacked area plot for purposes
    ax2.stackplot(times, purpose_percentages.T, labels=purpose_names, alpha=0.8)
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Population Percentage (%)")
    ax2.set_title("Population Activity Purpose Patterns")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path("saved_models/mode_generative_ode_batched") / f"population_patterns_{population_size}_people.png"
    plt.savefig(save_path)
    print(f"   📈 Population patterns plot saved to '{save_path}'")
    plt.close()

def demo_custom_scenarios():
    """Demonstrate custom scenario generation."""
    print("\n🎯 Demo 4: Custom Scenario Generation")
    print("=" * 50)
    
    scenarios = [
        {"name": "Rush Hour Analysis", "people": 50, "times": np.linspace(7, 9, 20)},
        {"name": "Evening Activity", "people": 30, "times": np.linspace(17, 22, 25)},
        {"name": "Weekend Pattern", "people": 20, "times": np.linspace(10, 16, 30)}
    ]
    
    model_path = "saved_models/mode_generative_ode_batched/latent_ode_best_model_batched.pth"
    
    try:
        engine = BatchedInferenceEngine(model_path)
        
        for scenario in scenarios:
            print(f"\n📋 Scenario: {scenario['name']}")
            
            # Generate person IDs
            person_ids = [1 + (i % 2) for i in range(scenario['people'])]
            times_tensor = engine.device.tensor if hasattr(engine, 'device') else torch.tensor
            import torch
            custom_times = torch.tensor(scenario['times'], dtype=torch.float32).to(engine.device)
            
            # Run inference
            predictions = engine.batch_inference(person_ids, custom_times, batch_size=32)
            
            # Quick analysis
            pred_modes = torch.argmax(predictions['mode_logits'], dim=-1).cpu().numpy()
            mode_names = ["Stay", "Walk", "Car", "Public_Transit"]
            
            # Calculate mode distribution
            mode_dist = {}
            for mode_id, mode_name in enumerate(mode_names):
                count = np.sum(pred_modes == mode_id)
                percentage = count / pred_modes.size * 100
                mode_dist[mode_name] = percentage
            
            print(f"   Mode distribution: {mode_dist}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🎮 Batched Inference Demo for Generative Latent ODE")
    print("🎯 This demo showcases scalable mobility prediction")
    print("=" * 80)
    
    # Run all demos
    demo_quick_inference()
    demo_scalability_test()
    demo_large_population_simulation()
    demo_custom_scenarios()
    
    print("\n" + "=" * 80)
    print("🎉 Demo complete! The batched inference system can now handle")
    print("   thousands of people efficiently for real-world deployment.")
    print("🚀 Ready for city-scale mobility simulation!") 
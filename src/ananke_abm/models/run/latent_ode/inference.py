"""
Scalable inference module for the Generative Latent ODE model.
Provides batched inference capabilities for processing thousands of people efficiently.
"""
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

from ananke_abm.models.run.latent_ode.config import GenerativeODEConfig
from ananke_abm.models.run.latent_ode.data import DataProcessor
from ananke_abm.models.run.latent_ode.model import GenerativeODE

class BatchedInferenceEngine:
    """
    High-performance batched inference engine for the Generative ODE model.
    Designed to process thousands of people simultaneously.
    """
    
    def __init__(self, model_path: str, config: Optional[GenerativeODEConfig] = None, device: str = "auto"):
        """
        Initialize the batched inference engine.
        
        Args:
            model_path: Path to trained model weights
            config: Model configuration (uses default if None)
            device: Device to use ("auto", "cpu", "cuda")
        """
        self.config = config or GenerativeODEConfig()
        
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"🔬 Inference engine using device: {self.device}")
        
        # Initialize data processor
        self.processor = DataProcessor(self.device, self.config)
        
        # Load model
        self._load_model(model_path)
        
    def _load_model(self, model_path: str):
        """Load the trained model."""
        # Get dimensions from sample data
        init_data = self.processor.get_data(person_id=1)
        
        self.model = GenerativeODE(
            person_feat_dim=init_data["person_features"].shape[-1],
            num_zone_features=init_data["all_zone_features"].shape[-1],
            config=self.config,
        ).to(self.device)
        
        # Load weights
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"✅ Model loaded from {model_path}")
        
        # Cache zone features for efficiency
        sample_data = self.processor.get_data(person_id=1)
        self.all_zone_features = sample_data["all_zone_features"]
        self.adjacency_matrix = sample_data["adjacency_matrix"]
        
    def batch_inference(self, person_ids: List[int], times: torch.Tensor, 
                       batch_size: int = 64) -> Dict[str, torch.Tensor]:
        """
        Perform batched inference on multiple people.
        
        Args:
            person_ids: List of person IDs to process
            times: Time points for trajectory generation [num_times]
            batch_size: Number of people to process simultaneously
            
        Returns:
            Dictionary with predictions for all people
        """
        all_predictions = {
            'location_logits': [],
            'purpose_logits': [], 
            'mode_logits': [],
            'person_names': []
        }
        
        num_people = len(person_ids)
        print(f"🚀 Starting batched inference for {num_people} people (batch_size={batch_size})")
        
        with torch.no_grad():
            for batch_start in range(0, num_people, batch_size):
                batch_end = min(batch_start + batch_size, num_people)
                current_batch_ids = person_ids[batch_start:batch_end]
                
                print(f"   Processing batch {batch_start//batch_size + 1}/{(num_people-1)//batch_size + 1}: people {batch_start+1}-{batch_end}")
                
                # Process current batch
                batch_predictions = self._process_batch(current_batch_ids, times)
                
                # Accumulate results
                for key in ['location_logits', 'purpose_logits', 'mode_logits']:
                    all_predictions[key].append(batch_predictions[key])
                all_predictions['person_names'].extend(batch_predictions['person_names'])
        
        # Concatenate all batches
        final_predictions = {}
        for key in ['location_logits', 'purpose_logits', 'mode_logits']:
            final_predictions[key] = torch.cat(all_predictions[key], dim=0)
        final_predictions['person_names'] = all_predictions['person_names']
        final_predictions['times'] = times
        
        print(f"✅ Batched inference complete for {num_people} people")
        return final_predictions
    
    def _process_batch(self, person_ids: List[int], times: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process a single batch of people."""
        current_batch_size = len(person_ids)
        
        # Collect batch data
        person_features_list = []
        home_zone_features_list = []
        work_zone_features_list = []
        start_purpose_ids = []
        person_names = []
        
        for person_id in person_ids:
            data = self.processor.get_data(person_id=person_id)
            person_features_list.append(data["person_features"])
            home_zone_features_list.append(data["home_zone_features"])
            work_zone_features_list.append(data["work_zone_features"])
            start_purpose_ids.append(data["start_purpose_id"])
            person_names.append(data["person_name"])
        
        # Stack into batch tensors
        person_features_batch = torch.stack(person_features_list)
        home_zone_features_batch = torch.stack(home_zone_features_list)
        work_zone_features_batch = torch.stack(work_zone_features_list)
        start_purpose_id_batch = torch.tensor(start_purpose_ids, dtype=torch.long, device=self.device)
        
        # Single forward pass for entire batch
        pred_y_logits, _, pred_purpose_logits, pred_mode_logits, _, _ = self.model(
            person_features_batch,
            home_zone_features_batch, 
            work_zone_features_batch,
            start_purpose_id_batch,
            times,
            self.all_zone_features,
            self.adjacency_matrix
        )
        
        return {
            'location_logits': pred_y_logits,
            'purpose_logits': pred_purpose_logits,
            'mode_logits': pred_mode_logits,
            'person_names': person_names
        }
    
    def predict_trajectories(self, person_ids: List[int], time_resolution: int = 100, 
                           batch_size: int = 64) -> Dict[str, np.ndarray]:
        """
        Generate predicted trajectories for multiple people.
        
        Args:
            person_ids: List of person IDs to process
            time_resolution: Number of time points (0-24 hours)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with trajectory predictions as numpy arrays
        """
        times = torch.linspace(0, 24, time_resolution).to(self.device)
        
        # Get predictions
        predictions = self.batch_inference(person_ids, times, batch_size)
        
        # Convert to discrete predictions
        pred_locations = torch.argmax(predictions['location_logits'], dim=-1).cpu().numpy()
        pred_purposes = torch.argmax(predictions['purpose_logits'], dim=-1).cpu().numpy()
        pred_modes = torch.argmax(predictions['mode_logits'], dim=-1).cpu().numpy()
        
        return {
            'times': times.cpu().numpy(),
            'locations': pred_locations,  # [num_people, num_times]
            'purposes': pred_purposes,    # [num_people, num_times]
            'modes': pred_modes,         # [num_people, num_times]
            'person_names': predictions['person_names']
        }
    
    def benchmark_performance(self, num_people_list: List[int] = [1, 10, 50, 100], 
                            batch_size: int = 64, time_resolution: int = 100):
        """
        Benchmark inference performance at different scales.
        
        Args:
            num_people_list: List of population sizes to test
            batch_size: Batch size for processing
            time_resolution: Number of time points
        """
        print("🏁 Performance Benchmarking")
        print("=" * 50)
        
        # Use available person IDs (cycling if needed)
        available_ids = [1, 2]
        
        for num_people in num_people_list:
            # Create person ID list (cycling through available IDs)
            person_ids = [available_ids[i % len(available_ids)] for i in range(num_people)]
            
            # Benchmark
            start_time = time.time()
            predictions = self.predict_trajectories(person_ids, time_resolution, batch_size)
            end_time = time.time()
            
            # Calculate metrics
            inference_time = end_time - start_time
            people_per_second = num_people / inference_time
            time_per_person = inference_time / num_people
            
            print(f"📊 {num_people:4d} people: {inference_time:6.2f}s total | {people_per_second:8.1f} people/s | {time_per_person*1000:6.1f}ms per person")
        
        print("=" * 50)
        
        # Project to 1M people
        if len(num_people_list) >= 2:
            # Estimate performance for 1M people based on largest test
            largest_test = max(num_people_list)
            largest_time = None
            
            # Re-run largest test for accurate timing
            person_ids = [available_ids[i % len(available_ids)] for i in range(largest_test)]
            start_time = time.time()
            self.predict_trajectories(person_ids, time_resolution, batch_size)
            largest_time = time.time() - start_time
            
            projected_time_1m = (largest_time / largest_test) * 1_000_000
            projected_hours = projected_time_1m / 3600
            
            print(f"📈 Projected time for 1M people: {projected_time_1m:.0f}s ({projected_hours:.1f} hours)")

# Convenience function for quick inference
def quick_inference(person_ids: List[int], model_path: str = "saved_models/mode_generative_ode_batched/latent_ode_best_model_batched.pth",
                   batch_size: int = 64, time_resolution: int = 100) -> Dict[str, np.ndarray]:
    """
    Quick inference function for immediate use.
    
    Args:
        person_ids: List of person IDs to process
        model_path: Path to trained model
        batch_size: Batch size for processing
        time_resolution: Number of time points
        
    Returns:
        Dictionary with trajectory predictions
    """
    engine = BatchedInferenceEngine(model_path)
    return engine.predict_trajectories(person_ids, time_resolution, batch_size) 
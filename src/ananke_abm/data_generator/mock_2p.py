# create mock_2p.py - Two person scenario for testing model generalization
import torch
import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .mock_1p import create_mock_zone_graph  # Reuse the same zone structure

@dataclass
class Person:
    """Mock person with realistic attributes"""
    person_id: int
    name: str
    
    # Demographics
    age: int
    income: float  # Annual income in USD
    employment_status: str
    occupation: str
    
    # Behavioral attributes
    commute_preference: str  # car, public_transit, bike, walk
    activity_flexibility: float  # 0=rigid schedule, 1=very flexible
    social_tendency: float  # 0=homebody, 1=very social
    
    # Household attributes
    household_income: float
    household_size: int
    dwelling_type: str
    has_car: bool
    
    # Home location (zone_id where they live)
    home_zone: int
    work_zone: int

def create_sarah():
    """Create Sarah - Tech worker with car, regular schedule"""
    return Person(
        person_id=1,
        name="Sarah Chen",
        
        # Demographics
        age=32,
        income=75000,
        employment_status="full_time",
        occupation="software_engineer",
        
        # Behavioral attributes
        commute_preference="car",
        activity_flexibility=0.3,  # Fairly rigid schedule
        social_tendency=0.6,  # Moderately social
        
        # Household attributes
        household_income=75000,
        household_size=1,
        dwelling_type="apartment",
        has_car=True,
        
        # Location
        home_zone=1,  # Riverside Apartments
        work_zone=5   # Tech Business Park
    )

def create_marcus():
    """Create Marcus - Retail worker with public transit, flexible schedule"""
    return Person(
        person_id=2,
        name="Marcus Rodriguez",
        
        # Demographics - Different from Sarah
        age=26,  # Younger
        income=35000,  # Lower income
        employment_status="part_time",  # Different employment
        occupation="retail_assistant",
        
        # Behavioral attributes - Contrasting with Sarah
        commute_preference="public_transit",  # No car
        activity_flexibility=0.8,  # Very flexible schedule
        social_tendency=0.9,  # Very social
        
        # Household attributes
        household_income=35000,
        household_size=1,
        dwelling_type="shared_house",
        has_car=False,  # No car
        
        # Location - Different zones
        home_zone=3,  # Downtown Residential (high transit access)
        work_zone=6   # Grand Mall (retail job)
    )

def create_sarah_daily_pattern():
    """Sarah's regular tech worker routine (reuse from original)"""
    
    # Time format: hours from midnight (0-24)
    daily_schedule = [
        {"time": 0.0, "zone": 1, "activity": "sleep", "description": "At home sleeping"},
        {"time": 7.0, "zone": 1, "activity": "morning_routine", "description": "Wake up, breakfast"},
        {"time": 7.5, "zone": 1, "activity": "prepare_commute", "description": "Getting ready to leave"},
        
        # Morning commute: Home → Work (via shopping plaza)
        {"time": 8.0, "zone": 1, "activity": "start_commute", "description": "Leaving home"},
        {"time": 8.13, "zone": 2, "activity": "transit", "description": "Passing through shopping area"},
        {"time": 8.27, "zone": 3, "activity": "transit", "description": "Through downtown residential"},
        {"time": 8.35, "zone": 4, "activity": "transit", "description": "Through entertainment district"},
        {"time": 8.45, "zone": 5, "activity": "arrive_work", "description": "Arriving at work"},
        
        # Work day
        {"time": 9.0, "zone": 5, "activity": "work", "description": "Morning work"},
        
        # Lunch break: Work → Mall → Work  
        {"time": 12.0, "zone": 5, "activity": "lunch_start", "description": "Leaving for lunch"},
        {"time": 12.08, "zone": 6, "activity": "lunch", "description": "Lunch at mall"},
        {"time": 13.0, "zone": 6, "activity": "lunch_end", "description": "Finishing lunch"},
        {"time": 13.08, "zone": 5, "activity": "work", "description": "Back to work"},
        
        # Afternoon work
        {"time": 17.0, "zone": 5, "activity": "end_work", "description": "Leaving work"},
        
        # Evening: Work → Gym → Home
        {"time": 17.15, "zone": 4, "activity": "transit", "description": "Through entertainment district"},
        {"time": 17.4, "zone": 7, "activity": "gym", "description": "Evening workout"},
        {"time": 19.0, "zone": 7, "activity": "gym_end", "description": "Leaving gym"},
        {"time": 19.08, "zone": 2, "activity": "transit", "description": "Passing shopping area"},
        {"time": 19.17, "zone": 1, "activity": "arrive_home", "description": "Back home"},
        
        # Evening at home
        {"time": 19.5, "zone": 1, "activity": "dinner", "description": "Dinner at home"},
        {"time": 21.0, "zone": 1, "activity": "evening", "description": "Relaxing at home"},
        {"time": 23.0, "zone": 1, "activity": "sleep", "description": "Going to sleep"},
    ]
    
    return daily_schedule

def create_marcus_daily_pattern():
    """Marcus's flexible retail worker routine - very different from Sarah"""
    
    # Marcus has a late shift, uses public transit, more social activities
    # Lives in downtown, works at mall, much more flexible and social
    
    daily_schedule = [
        {"time": 0.0, "zone": 3, "activity": "sleep", "description": "At home sleeping"},
        {"time": 9.0, "zone": 3, "activity": "wake_up", "description": "Late wake up"},
        {"time": 9.5, "zone": 3, "activity": "morning_routine", "description": "Breakfast, getting ready"},
        
        # Late morning: Home → Park (exercise/relaxation)
        {"time": 10.0, "zone": 3, "activity": "leaving_home", "description": "Starting day"},
        {"time": 10.08, "zone": 4, "activity": "transit", "description": "Through entertainment district"},
        {"time": 10.2, "zone": 7, "activity": "transit", "description": "Passing fitness complex"},
        {"time": 10.27, "zone": 8, "activity": "exercise", "description": "Morning jog in park"},
        
        # Mid-morning social time: Park → Entertainment District
        {"time": 11.5, "zone": 8, "activity": "leaving_park", "description": "Finished exercising"},
        {"time": 11.57, "zone": 7, "activity": "transit", "description": "Through fitness complex"},
        {"time": 12.1, "zone": 4, "activity": "social", "description": "Meeting friends for coffee"},
        
        # Afternoon: Entertainment → Work (Mall)
        {"time": 13.5, "zone": 4, "activity": "leaving_social", "description": "Heading to work"},
        {"time": 13.65, "zone": 5, "activity": "transit", "description": "Through business park"},
        {"time": 13.73, "zone": 6, "activity": "arrive_work", "description": "Starting retail shift"},
        
        # Afternoon work shift
        {"time": 14.0, "zone": 6, "activity": "work", "description": "Retail work"},
        
        # Short break during work
        {"time": 16.5, "zone": 6, "activity": "break", "description": "Work break"},
        {"time": 17.0, "zone": 6, "activity": "work", "description": "Back to work"},
        
        # Evening work continues
        {"time": 20.0, "zone": 6, "activity": "end_work", "description": "Finishing work shift"},
        
        # Evening social: Mall → Entertainment District (dinner/socializing)
        {"time": 20.17, "zone": 5, "activity": "transit", "description": "Through business park"},
        {"time": 20.25, "zone": 4, "activity": "dinner_social", "description": "Dinner with friends"},
        
        # Late evening: Entertainment → Home
        {"time": 22.0, "zone": 4, "activity": "leaving_social", "description": "Heading home"},
        {"time": 22.08, "zone": 3, "activity": "arrive_home", "description": "Back home"},
        
        # Late night at home
        {"time": 22.5, "zone": 3, "activity": "evening", "description": "Relaxing at home"},
        {"time": 24.0, "zone": 3, "activity": "sleep", "description": "Going to sleep (late)"},
    ]
    
    return daily_schedule

def create_training_data_single_person(person, schedule, zone_graph):
    """Convert single person schedule to training format"""
    
    # Extract zone observations and times
    times = torch.tensor([event["time"] for event in schedule], dtype=torch.float32)
    zones = torch.tensor([event["zone"] - 1 for event in schedule], dtype=torch.long)  # Convert to 0-indexed
    
    # Person attributes as feature vector
    person_attrs = torch.tensor([
        person.age / 100.0,  # Normalize age
        person.income / 100000.0,  # Normalize income
        1.0 if person.employment_status == "full_time" else 0.0,
        1.0 if person.commute_preference == "car" else 0.0,
        person.activity_flexibility,
        person.social_tendency,
        person.household_size / 10.0,  # Normalize
        1.0 if person.has_car else 0.0
    ], dtype=torch.float32)
    
    # Zone graph as PyTorch Geometric format - convert to 0-indexed
    edge_list = [(u-1, v-1) for u, v in zone_graph.edges()]  # Convert to 0-indexed
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Zone features - reorder by zone_id to match 0-indexing
    zone_features = []
    for zone_id in sorted(zone_graph.nodes()):
        zone_data = zone_graph.nodes[zone_id]
        features = [
            zone_data["population"] / 10000.0,  # Normalize
            zone_data["job_opportunities"] / 5000.0,
            zone_data["retail_accessibility"],
            zone_data["transit_accessibility"], 
            zone_data["attractiveness"],
            zone_data["coordinates"][0] / 5.0,  # Normalize coordinates
            zone_data["coordinates"][1] / 5.0,
        ]
        zone_features.append(features)
    
    zone_features = torch.tensor(zone_features, dtype=torch.float32)
    
    return {
        "person_attrs": person_attrs,
        "times": times,
        "zone_observations": zones,  # Now 0-indexed
        "zone_features": zone_features,
        "edge_index": edge_index,
        "num_zones": len(zone_graph.nodes()),
        "person_name": person.name,
        "person_id": person.person_id
    }

def create_two_person_training_data():
    """Create training data with both people for learning person-specific patterns
    
    This creates training data where models can learn to associate:
    - Person attributes → Behavioral patterns
    
    Models will be trained on both Sarah and Marcus, then tested on their ability
    to predict the correct behavioral pattern when given each person's attributes.
    """
    
    # Create the shared zone graph
    zone_graph, zone_data = create_mock_zone_graph()
    
    # Create both people
    sarah = create_sarah()
    marcus = create_marcus()
    
    # Create their schedules
    sarah_schedule = create_sarah_daily_pattern()
    marcus_schedule = create_marcus_daily_pattern()
    
    # Convert to training format
    sarah_data = create_training_data_single_person(sarah, sarah_schedule, zone_graph)
    marcus_data = create_training_data_single_person(marcus, marcus_schedule, zone_graph)
    
    # Create a "batch" format where we have multiple training examples
    # This allows models to learn the mapping: person_attrs → trajectory_pattern
    
    # We'll create the training data as separate examples that can be used in batch training
    # Format compatible with existing training loops
    
    # Option 1: Use Sarah's data as the primary training format (with all person metadata)
    # This maintains compatibility while providing both people's data for evaluation
    
    combined_data = {
        # Primary training format (using Sarah's data structure)
        "person_attrs": sarah_data["person_attrs"],  # [8] - Sarah's attrs for compatibility
        "times": sarah_data["times"],  # [sarah_timepoints] - Sarah's times  
        "zone_observations": sarah_data["zone_observations"],  # [sarah_timepoints] - Sarah's zones
        "zone_features": sarah_data["zone_features"],  # Same for both
        "edge_index": sarah_data["edge_index"],  # Same for both
        "num_zones": sarah_data["num_zones"],
        
        # Additional data for both people
        "person_names": [sarah.name, marcus.name],
        "person_ids": [sarah.person_id, marcus.person_id],
        "sarah_data": sarah_data,  # Complete Sarah data
        "marcus_data": marcus_data,  # Complete Marcus data
        
        # For training frameworks that can handle multiple people
        "both_people_data": [sarah_data, marcus_data],
        "is_multi_person": True
    }
    
    return combined_data, sarah, marcus

def create_alternating_training_data():
    """Create training data that alternates between Sarah and Marcus observations
    
    This creates a single sequence that interleaves observations from both people,
    allowing models to learn person-specific patterns in a mixed training regime.
    """
    
    # Get individual data
    combined_data, sarah, marcus = create_two_person_training_data()
    sarah_data = combined_data['sarah_data']
    marcus_data = combined_data['marcus_data']
    
    # Create alternating sequence - we'll take turns between Sarah and Marcus
    # This creates a mixed training signal where person attributes vary per timestep
    
    alternating_observations = []
    person_attrs_per_timestep = []
    
    # Simple alternating pattern: Sarah, Marcus, Sarah, Marcus, etc.
    max_length = max(len(sarah_data["times"]), len(marcus_data["times"]))
    
    for i in range(max_length):
        # Add Sarah's observation if available
        if i < len(sarah_data["times"]):
            alternating_observations.append({
                "time": sarah_data["times"][i],
                "zone": sarah_data["zone_observations"][i],
                "person_attrs": sarah_data["person_attrs"],
                "person_name": "Sarah"
            })
            
        # Add Marcus's observation if available  
        if i < len(marcus_data["times"]):
            alternating_observations.append({
                "time": marcus_data["times"][i], 
                "zone": marcus_data["zone_observations"][i],
                "person_attrs": marcus_data["person_attrs"],
                "person_name": "Marcus"
            })
    
    # Sort by time to create chronological sequence
    alternating_observations.sort(key=lambda x: x["time"])
    
    # Convert to tensor format
    alternating_times = torch.tensor([obs["time"] for obs in alternating_observations])
    alternating_zones = torch.tensor([obs["zone"] for obs in alternating_observations])
    alternating_person_attrs = torch.stack([obs["person_attrs"] for obs in alternating_observations])
    
    alternating_data = {
        "person_attrs": alternating_person_attrs,  # [total_timepoints, 8] - varies per timestep
        "times": alternating_times,  # [total_timepoints] - chronologically ordered
        "zone_observations": alternating_zones,  # [total_timepoints] - mixed people
        "zone_features": sarah_data["zone_features"],  # Same for both
        "edge_index": sarah_data["edge_index"],  # Same for both
        "num_zones": sarah_data["num_zones"],
        
        # Metadata
        "observation_metadata": alternating_observations,
        "sarah_data": sarah_data,
        "marcus_data": marcus_data,
        "is_alternating": True
    }
    
    return alternating_data, sarah, marcus

if __name__ == "__main__":
    # Test the two-person data generation
    print("🧪 Testing Two-Person Data Generation")
    print("="*50)
    
    # Test combined data
    combined_data, sarah, marcus = create_two_person_training_data()
    
    print(f"👥 People Created:")
    print(f"   Sarah: {sarah.name}, age {sarah.age}, {sarah.occupation}")
    print(f"          Income: ${sarah.income:,}, Car: {sarah.has_car}, Flexibility: {sarah.activity_flexibility}")
    print(f"   Marcus: {marcus.name}, age {marcus.age}, {marcus.occupation}")
    print(f"           Income: ${marcus.income:,}, Car: {marcus.has_car}, Flexibility: {marcus.activity_flexibility}")
    
    print(f"\n📊 Combined Data:")
    print(f"   Total time points: {len(combined_data['times'])}")
    print(f"   Sarah points: {combined_data['sarah_length']}")
    print(f"   Marcus points: {combined_data['marcus_length']}")
    print(f"   Zones: {combined_data['num_zones']}")
    print(f"   Person attributes shape: {combined_data['person_attrs'].shape}")
    
    print("\n✅ Two-person data generation working!") 
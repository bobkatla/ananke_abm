# create mock_2p.py - Two person scenario for testing model generalization
import torch
import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .mock_locations import create_mock_zone_graph  # Reuse the same zone structure

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
        # At home in the morning
        {"time": 0.0, "zone": 1, "activity": "sleep", "description": "At home sleeping", "importance": "anchor", "travel_mode": "Stay"},
        {"time": 7.0, "zone": 1, "activity": "morning_routine", "description": "Wake up, breakfast", "travel_mode": "Stay"},
        
        # Morning commute
        {"time": 7.5, "zone": 1, "activity": "transit", "description": "Leaving home for work", "travel_mode": "Car"},
        {"time": 8.45, "zone": 5, "activity": "work", "description": "Arriving at work", "travel_mode": "Stay"},

        # Lunch break
        {"time": 12.0, "zone": 5, "activity": "transit", "description": "Leaving for lunch", "travel_mode": "Walk"},
        {"time": 12.08, "zone": 6, "activity": "lunch", "description": "Lunch at mall", "travel_mode": "Stay"},
        {"time": 13.0, "zone": 6, "activity": "transit", "description": "Returning to work", "travel_mode": "Walk"},
        {"time": 13.08, "zone": 5, "activity": "work", "description": "Back to work", "travel_mode": "Stay"},

        # Evening commute and activity
        {"time": 17.0, "zone": 5, "activity": "transit", "description": "Leaving work for gym", "travel_mode": "Car"},
        {"time": 17.4, "zone": 7, "activity": "gym", "description": "Evening workout", "travel_mode": "Stay"},
        {"time": 19.0, "zone": 7, "activity": "transit", "description": "Leaving gym for home", "travel_mode": "Car"},
        {"time": 19.17, "zone": 1, "activity": "arrive_home", "description": "Back home", "travel_mode": "Stay"},
        
        # Evening at home
        {"time": 19.5, "zone": 1, "activity": "dinner", "description": "Dinner at home", "travel_mode": "Stay"},
        {"time": 21.0, "zone": 1, "activity": "evening", "description": "Relaxing at home", "travel_mode": "Stay"},
        {"time": 23.0, "zone": 1, "activity": "sleep", "description": "Going to sleep", "travel_mode": "Stay"},
        {"time": 24.0, "zone": 1, "activity": "sleep", "description": "End of day at home", "importance": "anchor", "travel_mode": "Stay"},
    ]
    
    return daily_schedule

def create_marcus_daily_pattern():
    """Marcus's flexible retail worker routine - very different from Sarah"""
    
    # Marcus has a late shift, uses public transit, more social activities
    # Lives in downtown, works at mall, much more flexible and social
    
    daily_schedule = [
        # At home in the morning
        {"time": 0.0, "zone": 3, "activity": "sleep", "description": "At home sleeping", "importance": "anchor", "travel_mode": "Stay"},
        {"time": 9.5, "zone": 3, "activity": "morning_routine", "description": "Breakfast, getting ready", "travel_mode": "Stay"},
        
        # Late morning activity
        {"time": 10.0, "zone": 3, "activity": "transit", "description": "Leaving home for park", "travel_mode": "Walk"},
        {"time": 10.27, "zone": 8, "activity": "exercise", "description": "Morning jog in park", "travel_mode": "Stay"},

        # Mid-morning social time
        {"time": 11.5, "zone": 8, "activity": "transit", "description": "Leaving park for social", "travel_mode": "Walk"},
        {"time": 12.1, "zone": 4, "activity": "social", "description": "Meeting friends for coffee", "travel_mode": "Stay"},
        
        # Afternoon commute to work
        {"time": 13.5, "zone": 4, "activity": "transit", "description": "Heading to work", "travel_mode": "Public_Transit"},
        {"time": 13.73, "zone": 6, "activity": "work", "description": "Starting retail shift", "travel_mode": "Stay"},

        # Work shift
        {"time": 20.0, "zone": 6, "activity": "transit", "description": "Finishing work shift for dinner", "travel_mode": "Public_Transit"},
        {"time": 20.25, "zone": 4, "activity": "dinner_social", "description": "Dinner with friends", "travel_mode": "Stay"},
        
        # Late evening commute home
        {"time": 22.0, "zone": 4, "activity": "transit", "description": "Heading home", "travel_mode": "Public_Transit"},
        {"time": 22.08, "zone": 3, "activity": "arrive_home", "description": "Back home", "travel_mode": "Stay"},
        
        # Late night at home
        {"time": 22.5, "zone": 3, "activity": "evening", "description": "Relaxing at home", "travel_mode": "Stay"},
        {"time": 24.0, "zone": 3, "activity": "sleep", "description": "Going to sleep (late)", "importance": "anchor", "travel_mode": "Stay"},
    ]
    
    return daily_schedule

def create_training_data_single_person(
    person, 
    schedule, 
    zone_graph, 
    repeat_pattern=True, 
    num_days=14, 
    time_noise_std=0.1
):
    """
    Convert single person schedule to training format.
    If repeat_pattern is True, repeats the daily pattern with added noise 
    to create a more realistic long sequence.
    """
    
    all_times = []
    all_zones = []
    all_activities = []
    all_importances = []
    all_travel_modes = []
    
    if repeat_pattern:
        # Repeat the schedule for num_days to create a long sequence
        for day in range(num_days):
            day_offset = day * 24.0  # Add 24 hours for each new day
            
            for event in schedule:
                # Add random noise to the event time for more realistic data
                time_noise = np.random.normal(0, time_noise_std) if event['time'] > 0 else 0
                
                # Ensure time is always increasing
                new_time = event["time"] + day_offset + time_noise
                if len(all_times) > 0 and new_time <= all_times[-1]:
                    new_time = all_times[-1] + 0.01 # Add small delta

                all_times.append(new_time)
                all_zones.append(event["zone"] - 1) # 0-indexed
                all_activities.append(event["activity"])
                all_importances.append(event.get("importance", "normal"))
                all_travel_modes.append(event.get("travel_mode", "Stay"))
    else:
        # Original behavior: process the schedule once without repetition or noise
        for event in schedule:
            all_times.append(event["time"])
            all_zones.append(event["zone"] - 1)
            all_activities.append(event["activity"])
            all_importances.append(event.get("importance", "normal"))
            all_travel_modes.append(event.get("travel_mode", "Stay"))

    # Extract zone observations and times
    times = torch.tensor(all_times, dtype=torch.float32)
    zones = torch.tensor(all_zones, dtype=torch.long)
    
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
        "activities": all_activities,
        "importances": all_importances,
        "travel_modes": all_travel_modes,
        "zone_features": zone_features,
        "edge_index": edge_index,
        "num_zones": len(zone_graph.nodes()),
        "person_name": person.name,
        "person_id": person.person_id,
        "home_zone_id": person.home_zone - 1, # 0-indexed
        "work_zone_id": person.work_zone - 1  # 0-indexed
    }

def create_two_person_training_data(repeat_pattern=True):
    """Create training data with both people for learning person-specific patterns.
    
    Args:
        repeat_pattern (bool): If True, generates a longer, noisy sequence. 
                               If False, generates a single, clean daily pattern.
    """
    
    # Create the shared zone graph
    zone_graph, zone_data = create_mock_zone_graph()
    
    # --- Sarah ---
    sarah = create_sarah()
    sarah_schedule = create_sarah_daily_pattern()
    sarah_data = create_training_data_single_person(sarah, sarah_schedule, zone_graph, repeat_pattern=repeat_pattern)
    
    # --- Marcus ---
    marcus = create_marcus()
    marcus_schedule = create_marcus_daily_pattern()
    marcus_data = create_training_data_single_person(marcus, marcus_schedule, zone_graph, repeat_pattern=repeat_pattern)
    
    return sarah_data, marcus_data

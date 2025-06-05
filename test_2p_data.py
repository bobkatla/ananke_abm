#!/usr/bin/env python3
"""
Test the two-person data generation
"""

import sys
import os
sys.path.insert(0, 'src')

from ananke_abm.data_generator.mock_2p import create_two_person_training_data

if __name__ == "__main__":
    print("🧪 Testing Two-Person Data Generation")
    print("="*50)
    
    # Test data generation
    combined_data, sarah, marcus = create_two_person_training_data()
    
    print(f"👥 People Created:")
    print(f"   Sarah: {sarah.name}, age {sarah.age}, {sarah.occupation}")
    print(f"          Income: ${sarah.income:,}, Car: {sarah.has_car}, Flexibility: {sarah.activity_flexibility}")
    print(f"          Home: Zone {sarah.home_zone}, Work: Zone {sarah.work_zone}")
    
    print(f"   Marcus: {marcus.name}, age {marcus.age}, {marcus.occupation}")
    print(f"           Income: ${marcus.income:,}, Car: {marcus.has_car}, Flexibility: {marcus.activity_flexibility}")
    print(f"           Home: Zone {marcus.home_zone}, Work: Zone {marcus.work_zone}")
    
    print(f"\n📊 Combined Data:")
    print(f"   Primary format time points: {len(combined_data['times'])}")
    print(f"   Sarah points: {len(combined_data['sarah_data']['times'])}")
    print(f"   Marcus points: {len(combined_data['marcus_data']['times'])}")
    print(f"   Zones: {combined_data['num_zones']}")
    print(f"   Person attributes shape: {combined_data['person_attrs'].shape}")
    print(f"   Multi-person training: {combined_data['is_multi_person']}")
    
    # Show first few observations from each person
    print(f"\n📝 Sample Schedules:")
    print(f"   Sarah's first 5 time points:")
    sarah_data = combined_data['sarah_data']
    for i in range(5):
        t = sarah_data['times'][i]
        z = sarah_data['zone_observations'][i] + 1  # Convert back to 1-indexed
        print(f"     t={t:4.1f}h: Zone {z}")
    
    print(f"   Marcus's first 5 time points:")
    marcus_data = combined_data['marcus_data']
    for i in range(5):
        t = marcus_data['times'][i]
        z = marcus_data['zone_observations'][i] + 1  # Convert back to 1-indexed
        print(f"     t={t:4.1f}h: Zone {z}")
    
    print("\n✅ Two-person data generation successful!")
    
    # Show the key differences
    print(f"\n🔍 Key Contrasts:")
    print(f"   Age: Sarah {sarah.age} vs Marcus {marcus.age}")
    print(f"   Income: Sarah ${sarah.income:,} vs Marcus ${marcus.income:,}")
    print(f"   Transport: Sarah {sarah.commute_preference} vs Marcus {marcus.commute_preference}")
    print(f"   Flexibility: Sarah {sarah.activity_flexibility} vs Marcus {marcus.activity_flexibility}")
    print(f"   Social: Sarah {sarah.social_tendency} vs Marcus {marcus.social_tendency}")
    print(f"   Schedule: Sarah starts {sarah_data['times'][1]:.1f}h, Marcus starts {marcus_data['times'][1]:.1f}h") 
#!/usr/bin/env python3
"""Analyze transition patterns in the mock data"""

import sys
from collections import Counter
sys.path.append('src/ananke_abm/data_generator')

from ananke_abm.data_generator.mock_2p import create_sarah_daily_pattern, create_marcus_daily_pattern

def analyze_transitions(schedule, person_name):
    print(f'\n=== {person_name.upper()} TRANSITION ANALYSIS ===')
    
    print("\nFull sequence:")
    for i, event in enumerate(schedule):
        print(f"T{i:2d}: {event['activity']:15s} + {event['travel_mode']:15s} (Zone {event['zone']})")
    
    print(f"\nTransition patterns:")
    forbidden_combinations = []
    
    for i in range(len(schedule)-1):
        curr = schedule[i]
        next_event = schedule[i+1]
        
        # Check for forbidden combination: "transit" activity + "Stay" mode
        if curr['activity'] == 'transit' and curr['travel_mode'] == 'Stay':
            forbidden_combinations.append(f"T{i}: transit + Stay")
        
        # Check transition
        activity_change = curr['activity'] != next_event['activity'] 
        mode_change = curr['travel_mode'] != next_event['travel_mode']
        zone_change = curr['zone'] != next_event['zone']
        
        transition_type = ""
        if zone_change and activity_change and mode_change:
            transition_type = " [COMPLEX]"
        elif zone_change:
            transition_type = " [LOCATION_CHANGE]"
        elif activity_change:
            transition_type = " [ACTIVITY_CHANGE]"
        elif mode_change:
            transition_type = " [MODE_CHANGE]"
            
        print(f"T{i:2d}→{i+1:2d}: {curr['activity']:12s}+{curr['travel_mode']:12s} → {next_event['activity']:12s}+{next_event['travel_mode']:12s}{transition_type}")
    
    if forbidden_combinations:
        print(f"\n❌ FORBIDDEN COMBINATIONS FOUND:")
        for combo in forbidden_combinations:
            print(f"   {combo}")
    else:
        print(f"\n✅ No forbidden combinations found")
    
    return forbidden_combinations

# Analyze both schedules
sarah_schedule = create_sarah_daily_pattern()
marcus_schedule = create_marcus_daily_pattern()

sarah_issues = analyze_transitions(sarah_schedule, "Sarah")
marcus_issues = analyze_transitions(marcus_schedule, "Marcus")

# Summary analysis
print(f"\n=== SUMMARY ANALYSIS ===")

# Count mode and activity distributions
sarah_modes = [event['travel_mode'] for event in sarah_schedule]
sarah_activities = [event['activity'] for event in sarah_schedule]
marcus_modes = [event['travel_mode'] for event in marcus_schedule]
marcus_activities = [event['activity'] for event in marcus_schedule]

print(f"\nMode distributions:")
print(f"Sarah: {dict(Counter(sarah_modes))}")
print(f"Marcus: {dict(Counter(marcus_modes))}")

print(f"\nActivity distributions:")
print(f"Sarah: {dict(Counter(sarah_activities))}")
print(f"Marcus: {dict(Counter(marcus_activities))}")

# Check for oversmoothing risk
total_stay = sarah_modes.count('Stay') + marcus_modes.count('Stay')
total_transit = sarah_activities.count('transit') + marcus_activities.count('transit')
total_events = len(sarah_schedule) + len(marcus_schedule)

print(f"\n🚨 OVERSMOOTHING RISK ANALYSIS:")
print(f"   'Stay' mode: {total_stay}/{total_events} = {total_stay/total_events:.1%}")
print(f"   'transit' activity: {total_transit}/{total_events} = {total_transit/total_events:.1%}")
print(f"   Combined dominance: {(total_stay + total_transit)/total_events:.1%}")

if (total_stay + total_transit)/total_events > 0.5:
    print(f"   ⚠️  HIGH RISK: Stay+Transit > 50% of training data")
else:
    print(f"   ✅ MODERATE RISK: Stay+Transit < 50% of training data") 
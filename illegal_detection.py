import pandas as pd
import numpy as np
import json
import pickle
from geopy.distance import geodesic
from datetime import datetime
with open('data/legal_mines.json', 'r') as f:
    data = json.load(f)
    legal_mines = data['legal_mines']

xgb_model = pickle.load(open('ml/car_model.pkl', 'rb'))
iso_model  = pickle.load(open('ml/isolation_model.pkl', 'rb'))

def is_night_time(hour=None):
    if hour is None:
        hour = datetime.now().hour
    return hour >= 22 or hour <= 5

def estimate_distance(car_drop, vibration_strength):
    if vibration_strength > 0.8 and car_drop > 0.3:
        return "0-500m — VERY CLOSE"
    elif vibration_strength > 0.5 and car_drop > 0.2:
        return "500m-1km — CLOSE"
    elif vibration_strength > 0.3 and car_drop > 0.1:
        return "1km-2km — MODERATE"
    else:
        return "2km+ — DISTANT"

def calculate_confidence(car_drop, vibration,
                          is_night, distance_km):
    score = 0
    if car_drop > 0.3:   score += 35
    elif car_drop > 0.2: score += 20
    elif car_drop > 0.1: score += 10

    if vibration:        score += 30
    if is_night:         score += 25
    if distance_km > 3:  score += 10

    return min(score, 99)

def check_illegal_mining(alert_zone_coords,
                          car_current,
                          car_previous,
                          vibration_detected,
                          timestamp=None):
    """
    Main illegal mining detection function

    Parameters:
    alert_zone_coords : tuple (lat, lon)
    car_current       : current CAR value
    car_previous      : CAR value 24 hours ago
    vibration_detected: bool
    timestamp         : datetime object

    Returns: dict with detection result
    """

    if timestamp is None:
        timestamp = datetime.now()

    car_drop = car_previous - car_current
    night    = is_night_time(timestamp.hour)

    nearest_mine     = None
    min_distance     = float('inf')
    nearest_radius   = 0
    nearest_operator = ""

    for mine_name, mine_data in legal_mines.items():
        mine_coords = (mine_data['lat'], mine_data['lon'])
        distance    = geodesic(
            alert_zone_coords, mine_coords
        ).km

        if distance < min_distance:
            min_distance     = distance
            nearest_mine     = mine_name
            nearest_radius   = mine_data['radius_km']
            nearest_operator = mine_data['operator']

    if min_distance <= nearest_radius:
        return {
            "status":      "LEGAL_MINING",
            "mine":        nearest_mine,
            "operator":    nearest_operator,
            "distance_km": round(min_distance, 2),
            "radius_km":   nearest_radius,
            "message":     (f"Zone within approved boundary of "
                           f"{nearest_mine} ({nearest_operator}). "
                           f"CAR degradation is expected."),
            "alert":       False
        }

    confidence = calculate_confidence(
        car_drop, vibration_detected,
        night, min_distance
    )

    if confidence < 30:
        return {
            "status":     "NATURAL_DEGRADATION",
            "confidence": confidence,
            "message":    ("CAR drop detected but insufficient "
                          "evidence for illegal mining. "
                          "Could be natural degradation."),
            "alert":      False
        }

    est_distance = estimate_distance(
        car_drop,
        0.8 if vibration_detected else 0.2
    )

    return {
        "status":           "ILLEGAL_MINING_DETECTED",
        "confidence":       confidence,
        "car_drop":         round(car_drop, 3),
        "car_current":      round(car_current, 3),
        "nearest_legal_mine": nearest_mine,
        "distance_to_nearest_legal": round(min_distance, 2),
        "estimated_source_distance": est_distance,
        "vibration_at_night": (vibration_detected and night),
        "timestamp":        timestamp.isoformat(),
        "alert":            True,
        "alert_level":      "CLASSIFIED",
        "actions": [
            "IMMEDIATE: Do not route convoys through this zone",
            "Deploy verification team to estimated source location",
            "Request satellite imagery of flagged coordinates",
            "Alert District Collector and local authorities",
            "Initiate legal proceedings if confirmed"
        ],
        "message": (
            f"ILLEGAL MINING SUSPECTED — {confidence}% confidence. "
            f"Unexpected CAR drop of {car_drop:.2f} detected. "
            f"Nearest legal mine ({nearest_mine}) is "
            f"{min_distance:.2f}km away — outside approved "
            f"boundary. Estimated source: {est_distance}."
        )
    }


if __name__ == "__main__":
    print("Testing Illegal Mining Detection...")
    print("=" * 50)

    # Test 1: Legal zone
    result1 = check_illegal_mining(
        alert_zone_coords=(24.1150, 82.6650),
        car_current=0.25,
        car_previous=0.65,
        vibration_detected=True
    )
    print(f"\nTest 1 (Near legal mine):")
    print(f"Status: {result1['status']}")
    print(f"Message: {result1['message']}")

    result2 = check_illegal_mining(
        alert_zone_coords=(24.1750, 82.7250),
        car_current=0.20,
        car_previous=0.65,
        vibration_detected=True,
        timestamp=datetime(2024, 3, 5, 3, 0, 0)
    )
    print(f"\nTest 2 (Suspicious zone — 3AM):")
    print(f"Status:     {result2['status']}")
    print(f"Confidence: {result2['confidence']}%")
    print(f"Message:    {result2['message']}")
    if result2.get('actions'):
        print("Actions:")
        for a in result2['actions']:
            print(f"  → {a}")
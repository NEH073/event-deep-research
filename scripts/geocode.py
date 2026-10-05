import json

# Input JSON file from current directory
import os
import time

import requests

INPUT_FILE = os.path.join(os.path.dirname(__file__), "events.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "events_with_coords.json")


# Geocoding function using OpenStreetMap Nominatim
def geocode_location(location_name: str):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": location_name, "format": "json", "limit": 1}
    headers = {"User-Agent": "EventGeocoder/1.0"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        print(f"Error geocoding {location_name}: {e}")

    return None, None


def main():
    # Load events
    with open(INPUT_FILE, encoding="utf-8") as f:
        events = json.load(f)

    # Process each event
    for i, event in enumerate(events, 1):
        # Add auto-incremental ID
        event["id"] = i

        # Only process location if it's not empty
        if isinstance(event.get("location"), str) and event["location"].strip():
            loc_name = event["location"]
            print(f"Geocoding: {loc_name} ...")
            lat, lng = geocode_location(loc_name)
            if lat and lng:
                event["location"] = {"name": loc_name, "lat": lat, "lng": lng}
            else:
                event["location"] = {"name": loc_name, "lat": None, "lng": None}

            # Be polite to the API
            time.sleep(1)
        else:
            # Remove location field entirely for empty locations
            if "location" in event:
                del event["location"]

    # Save new JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved updated events to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

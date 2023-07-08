import requests
import os
import json
from agent.tools import Tool
from typing import Union, Dict

BASE_URL = "http://dev.virtualearth.net/REST/V1/"


def get_coordinates(location: str, key):
    """Get the coordinates of a location"""
    url = BASE_URL + "Locations"
    params = {"query": location, "key": key}
    response = requests.get(url, params=params)
    json_data = response.json()
    coordinates = json_data["resourceSets"][0]["resources"][0]["point"]["coordinates"]
    return coordinates


class GetDistanceTool(Tool):
    def __init__(self, key):
        super().__init__()
        self.invoke_label = "GetDistance"
        self.key = key

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        # return "136.559 miles", 0, False, {}
        start, end = invoke_data.strip().split(",")
        url = (
            BASE_URL
            + "Routes/Driving?o=json&wp.0="
            + start
            + "&wp.1="
            + end
            + "&key="
            + self.key
        )
        # GET request
        r = requests.get(url)
        data = json.loads(r.text)
        # Extract route information
        route = data["resourceSets"][0]["resources"][0]
        # Extract distance in miles
        distance = route["travelDistance"]
        return distance, 0, False, {}

    def description(self) -> str:
        return (
            "GetDistance(start, end), get the distance between two locations in miles."
        )


class GetRouteTool(Tool):
    def __init__(self, key):
        super().__init__()
        self.invoke_label = "GetRoute"
        self.key = key

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        start, end = invoke_data.strip().split(",")
        url = (
            BASE_URL
            + "Routes/Driving?o=json&wp.0="
            + start
            + "&wp.1="
            + end
            + "&key="
            + self.key
        )
        # GET request
        r = requests.get(url)
        data = json.loads(r.text)
        # Extract route information
        route = data["resourceSets"][0]["resources"][0]
        itinerary = route["routeLegs"][0]["itineraryItems"]
        # Extract route text information
        route_text = []
        for item in itinerary:
            if "instruction" in item:
                route_text.append(item["instruction"]["text"])
        return route_text, 0, False, {}

    def description(self) -> str:
        return "GetRoute(start, end), get the route between two locations in miles."


class GetCoordinatesTool(Tool):
    def __init__(self, key):
        super().__init__()
        self.invoke_label = "GetCoordinates"
        self.key = key

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        coordinates = get_coordinates(invoke_data, self.key)
        return coordinates, 0, False, {}

    def description(self) -> str:
        return "GetCoordinates(location), get the coordinates of a location."


class SearchNearbyTool(Tool):
    def __init__(self, key):
        super().__init__()
        self.invoke_label = "SearchNearby"
        self.key = key

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        KEY = self.key
        search_term, latitude, longitude, places, radius = invoke_data.strip().split(
            ","
        )
        url = BASE_URL + "LocalSearch"
        if places != "unknown":
            latitude = get_coordinates(places, self.key)[0]
            longitude = get_coordinates(places, self.key)[1]
        # Build the request query string
        params = {
            "query": search_term,
            "userLocation": f"{latitude},{longitude}",
            "radius": radius,
            "key": KEY,
        }
        # Make the request
        response = requests.get(url, params=params)

        # Parse the response
        response_data = json.loads(response.content)

        # Get the results
        results = response_data["resourceSets"][0]["resources"]
        addresses = []
        for result in results:
            name = result["name"]
            address = result["Address"]["formattedAddress"]
            addresses.append(f"{name}: {address}")
        return addresses, 0, False, {}

    def description(self) -> str:
        return "SearchNearby(search_term, latitude, longitude, places, radius), search for places nearby a location, within a given radius, and return the results into a list."

#!/usr/bin/env python3
"""Summary"""
import requests


def availableShips(passengerCount):
    """Summary"""
    url = "https://swapi-api.hbtn.io/api/starships/"
    r = requests.get(url)
    json = r.json()
    result = json['results']
    ships = []

    while json["next"]:
        for res in result:
            if res["passengers"] == "n/a" or res["passengers"] == "unknown":
                continue
            if int(res["passengers"].replace(',', '')) >= passengerCount:
                ships.append(res["name"])
        url = json["next"]
        r = requests.get(url)
        json = r.json()
        results = json["results"]
    return ships

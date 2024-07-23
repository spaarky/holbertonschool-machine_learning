#!/usr/bin/env python3
"""Summary"""


def schools_by_topic(mongo_collection, topic):
    """Summary"""
    return mongo_collection.find({"topics": {"$in": [topic]}})

#!/usr/bin/env python3
"""Summary"""


def update_topics(mongo_collection, name, topics):
    """Summary"""
    search = {"name": name}
    new = {"$set": {"topics": topics}}

    mongo_collection.update_many(search, new)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for AI Marketplace Platform

Functions:
- haversine_distance: Calculate distance between two coordinates
- sort_by_distance: Sort items by distance from user location
- extract_location_query: Parse location from natural language query
"""

import math
import re
from typing import List, Dict, Any, Optional, Tuple


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    Uses the Haversine formula.

    Args:
        lat1, lon1: Latitude and longitude of point 1 (degrees)
        lat2, lon2: Latitude and longitude of point 2 (degrees)

    Returns:
        Distance in kilometers

    Example:
        >>> distance = haversine_distance(-6.2088, 106.8456, -6.9175, 107.6191)
        >>> print(f"{distance:.2f} km")  # Jakarta to Bandung
        126.78 km
    """
    # Earth radius in kilometers
    R = 6371.0

    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def sort_by_distance(
    items: List[Dict[str, Any]],
    user_lat: float,
    user_lon: float,
    lat_key: str = "latitude",
    lon_key: str = "longitude"
) -> List[Dict[str, Any]]:
    """
    Sort a list of items by distance from user location.
    Adds 'distance_km' field to each item.

    Args:
        items: List of dictionaries containing location data
        user_lat: User's latitude
        user_lon: User's longitude
        lat_key: Key name for latitude in items (default: "latitude")
        lon_key: Key name for longitude in items (default: "longitude")

    Returns:
        Sorted list with distance_km added to each item

    Example:
        >>> suppliers = [
        ...     {"name": "Supplier A", "latitude": -6.2, "longitude": 106.8},
        ...     {"name": "Supplier B", "latitude": -6.9, "longitude": 107.6}
        ... ]
        >>> sorted_suppliers = sort_by_distance(suppliers, -6.2088, 106.8456)
        >>> print(sorted_suppliers[0]["distance_km"])
    """
    for item in items:
        if lat_key in item and lon_key in item:
            item["distance_km"] = haversine_distance(
                user_lat, user_lon,
                item[lat_key], item[lon_key]
            )
        else:
            item["distance_km"] = float('inf')  # Put items without location at the end

    # Sort by distance
    sorted_items = sorted(items, key=lambda x: x["distance_km"])
    return sorted_items


def extract_location_query(query: str) -> Optional[str]:
    """
    Extract city/location from natural language query.

    Args:
        query: Natural language query string

    Returns:
        Extracted location string or None

    Example:
        >>> extract_location_query("laptop gaming di Jakarta")
        'Jakarta'
        >>> extract_location_query("cari laptop Jakarta Selatan")
        'Jakarta Selatan'
    """
    # Common Indonesian location patterns
    patterns = [
        r'\b(?:di|dekat|sekitar|area)\s+([A-Z][a-zA-Z\s]+?)(?:\s|$|,)',
        r'\b([A-Z][a-zA-Z\s]+?)\s+(?:Selatan|Utara|Timur|Barat|Pusat)\b',
        r'\b(Jakarta|Bandung|Surabaya|Medan|Semarang|Makassar|Palembang|Tangerang|Depok|Bekasi|Bogor)\b'
    ]

    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def format_price_idr(price: float) -> str:
    """
    Format price as Indonesian Rupiah.

    Args:
        price: Price in Rupiah

    Returns:
        Formatted string (e.g., "Rp 10.000.000")

    Example:
        >>> format_price_idr(10000000)
        'Rp 10.000.000'
    """
    # Format with thousand separators (dot for Indonesian style)
    price_str = f"{int(price):,}".replace(",", ".")
    return f"Rp {price_str}"


def build_ai_context(products: List[Dict[str, Any]], user_query: str) -> str:
    """
    Build context string for AI with product information.

    Args:
        products: List of product dictionaries with supplier info
        user_query: User's original query

    Returns:
        Formatted context string for AI prompt

    Example:
        >>> products = [{"name": "Laptop ASUS", "price": 10000000, ...}]
        >>> context = build_ai_context(products, "laptop gaming")
        >>> print(context)
    """
    if not products:
        return f"""User query: {user_query}

Available products: None found. Inform the user that no products match their criteria and suggest alternatives."""

    product_list = []
    for idx, prod in enumerate(products, 1):
        supplier_name = prod.get("supplier_name", "Unknown")
        distance = prod.get("distance_km", "N/A")
        distance_str = f"{distance:.1f} km" if isinstance(distance, (int, float)) else distance

        product_info = f"""{idx}. {prod['name']}
   - Price: {format_price_idr(prod['price'])}
   - Stock: {prod['stock_quantity']} units
   - Category: {prod.get('category', 'N/A')}
   - Supplier: {supplier_name} ({distance_str} from user)
   - Description: {prod.get('description', 'No description')[:100]}..."""

        product_list.append(product_info)

    context = f"""User query: {user_query}

Available products (sorted by distance):

{chr(10).join(product_list)}

Instructions:
1. Recommend products based on user's needs and budget
2. Prioritize nearby suppliers (lower distance_km)
3. Explain why each recommendation fits the user's requirements
4. Mention stock availability
5. Provide price comparison if multiple options exist
6. Use natural, conversational Indonesian or English based on user's language"""

    return context

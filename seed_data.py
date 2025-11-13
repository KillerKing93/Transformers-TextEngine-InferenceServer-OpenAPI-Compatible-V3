#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sample data seeding script for AI Marketplace Platform

Usage:
    python seed_data.py

This script populates the database with sample suppliers, products, and users
for testing and demonstration purposes.
"""

import os
import sys
from datetime import datetime

# Ensure we can import from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import SessionLocal, init_db
from models import Supplier, Product, User


def seed_database():
    """Seed the database with sample data."""

    print("[Seed] Initializing database...")
    init_db()

    db = SessionLocal()

    try:
        # Check if data already exists
        existing_suppliers = db.query(Supplier).count()
        if existing_suppliers > 0:
            print(f"[Seed] Database already has {existing_suppliers} suppliers. Skipping seed.")
            response = input("Do you want to clear and re-seed? (yes/no): ")
            if response.lower() != "yes":
                print("[Seed] Aborted.")
                return
            else:
                # Clear existing data
                print("[Seed] Clearing existing data...")
                db.query(Product).delete()
                db.query(Supplier).delete()
                db.query(User).delete()
                db.commit()
                print("[Seed] Cleared.")

        # Seed Suppliers
        print("[Seed] Creating suppliers...")

        suppliers_data = [
            {
                "name": "Budi Santoso",
                "business_name": "Toko Komputer Jakarta",
                "email": "budi@tokokomputer.com",
                "phone": "+62812345678",
                "address": "Jl. Sudirman No. 123, Jakarta Pusat",
                "latitude": -6.2088,
                "longitude": 106.8456,
                "city": "Jakarta",
                "province": "DKI Jakarta"
            },
            {
                "name": "Siti Rahma",
                "business_name": "Elektronik Bandung",
                "email": "siti@elektronikbdg.com",
                "phone": "+62856789012",
                "address": "Jl. Braga No. 45, Bandung",
                "latitude": -6.9175,
                "longitude": 107.6191,
                "city": "Bandung",
                "province": "Jawa Barat"
            },
            {
                "name": "Ahmad Fauzi",
                "business_name": "Tech Store Surabaya",
                "email": "ahmad@techstoresby.com",
                "phone": "+62877123456",
                "address": "Jl. Tunjungan No. 88, Surabaya",
                "latitude": -7.2575,
                "longitude": 112.7521,
                "city": "Surabaya",
                "province": "Jawa Timur"
            },
            {
                "name": "Dewi Lestari",
                "business_name": "Gadget Center Jakarta Selatan",
                "email": "dewi@gadgetcenter.com",
                "phone": "+62899887766",
                "address": "Jl. Fatmawati No. 200, Jakarta Selatan",
                "latitude": -6.2910,
                "longitude": 106.7968,
                "city": "Jakarta Selatan",
                "province": "DKI Jakarta"
            },
            {
                "name": "Rudi Hartono",
                "business_name": "Computer World Medan",
                "email": "rudi@computerworld.com",
                "phone": "+62811223344",
                "address": "Jl. Gatot Subroto No. 100, Medan",
                "latitude": 3.5952,
                "longitude": 98.6722,
                "city": "Medan",
                "province": "Sumatera Utara"
            }
        ]

        suppliers = []
        for data in suppliers_data:
            supplier = Supplier(**data)
            db.add(supplier)
            suppliers.append(supplier)

        db.commit()
        print(f"[Seed] Created {len(suppliers)} suppliers")

        # Seed Products
        print("[Seed] Creating products...")

        products_data = [
            # Jakarta - Toko Komputer
            {
                "supplier": suppliers[0],
                "name": "Laptop ASUS ROG Strix G15",
                "description": "Gaming laptop with RTX 4060, Ryzen 7 6800H, 16GB RAM, 512GB SSD",
                "price": 15999000,
                "stock_quantity": 5,
                "category": "laptop",
                "tags": "gaming,asus,ryzen,rtx",
                "sku": "ASU-ROG-G15-001"
            },
            {
                "supplier": suppliers[0],
                "name": "Monitor LG 27' 4K UHD",
                "description": "27 inch 4K monitor, IPS panel, HDR10 support",
                "price": 4500000,
                "stock_quantity": 10,
                "category": "monitor",
                "tags": "monitor,lg,4k,hdr",
                "sku": "LG-MON-27-4K"
            },
            {
                "supplier": suppliers[0],
                "name": "Keyboard Mechanical Logitech G Pro",
                "description": "TKL mechanical keyboard with GX Blue switches",
                "price": 1850000,
                "stock_quantity": 15,
                "category": "keyboard",
                "tags": "keyboard,logitech,mechanical,gaming",
                "sku": "LOG-KEY-GPR"
            },
            # Bandung - Elektronik
            {
                "supplier": suppliers[1],
                "name": "Laptop Lenovo ThinkPad X1 Carbon",
                "description": "Business laptop, Intel i7-1365U, 16GB RAM, 512GB SSD, 14 inch",
                "price": 22500000,
                "stock_quantity": 3,
                "category": "laptop",
                "tags": "business,lenovo,thinkpad,intel",
                "sku": "LEN-X1C-G11"
            },
            {
                "supplier": suppliers[1],
                "name": "Smartphone Samsung Galaxy S24",
                "description": "Flagship smartphone, Snapdragon 8 Gen 3, 8GB RAM, 256GB",
                "price": 13999000,
                "stock_quantity": 8,
                "category": "smartphone",
                "tags": "smartphone,samsung,flagship",
                "sku": "SAM-S24-256"
            },
            {
                "supplier": suppliers[1],
                "name": "Tablet iPad Air M2",
                "description": "11 inch iPad Air with M2 chip, 128GB WiFi",
                "price": 9499000,
                "stock_quantity": 6,
                "category": "tablet",
                "tags": "tablet,ipad,apple,m2",
                "sku": "APL-IPA-M2-128"
            },
            # Surabaya - Tech Store
            {
                "supplier": suppliers[2],
                "name": "Laptop HP Pavilion Gaming",
                "description": "Gaming laptop, Intel i5-12500H, RTX 3050, 16GB RAM, 512GB SSD",
                "price": 11999000,
                "stock_quantity": 7,
                "category": "laptop",
                "tags": "gaming,hp,intel,rtx3050",
                "sku": "HP-PAV-GAM-001"
            },
            {
                "supplier": suppliers[2],
                "name": "Mouse Logitech G502 HERO",
                "description": "Gaming mouse with 25K DPI sensor, customizable weights",
                "price": 550000,
                "stock_quantity": 20,
                "category": "mouse",
                "tags": "mouse,logitech,gaming",
                "sku": "LOG-G502-HERO"
            },
            {
                "supplier": suppliers[2],
                "name": "Printer Epson EcoTank L3250",
                "description": "All-in-one printer with WiFi, scanner, copier",
                "price": 2850000,
                "stock_quantity": 5,
                "category": "printer",
                "tags": "printer,epson,ecotank,wifi",
                "sku": "EPS-L3250-ECO"
            },
            # Jakarta Selatan - Gadget Center
            {
                "supplier": suppliers[3],
                "name": "Laptop MacBook Air M3",
                "description": "13 inch MacBook Air, M3 chip, 8GB RAM, 256GB SSD",
                "price": 17999000,
                "stock_quantity": 4,
                "category": "laptop",
                "tags": "laptop,apple,macbook,m3",
                "sku": "APL-MBA-M3-256"
            },
            {
                "supplier": suppliers[3],
                "name": "Smartphone iPhone 15 Pro",
                "description": "Pro smartphone with A17 Pro chip, 128GB, Titanium",
                "price": 19999000,
                "stock_quantity": 5,
                "category": "smartphone",
                "tags": "smartphone,iphone,apple,pro",
                "sku": "APL-IP15P-128"
            },
            {
                "supplier": suppliers[3],
                "name": "Monitor Samsung Odyssey G7",
                "description": "32 inch curved gaming monitor, 240Hz, 1440p, QLED",
                "price": 8999000,
                "stock_quantity": 3,
                "category": "monitor",
                "tags": "monitor,samsung,gaming,curved,240hz",
                "sku": "SAM-ODY-G7-32"
            },
            # Medan - Computer World
            {
                "supplier": suppliers[4],
                "name": "Laptop Acer Aspire 5",
                "description": "Budget laptop, Intel i5-1235U, 8GB RAM, 512GB SSD, 15.6 inch",
                "price": 7999000,
                "stock_quantity": 12,
                "category": "laptop",
                "tags": "laptop,acer,budget,intel",
                "sku": "ACR-ASP5-I5"
            },
            {
                "supplier": suppliers[4],
                "name": "Smartphone Xiaomi Redmi Note 13 Pro",
                "description": "Mid-range smartphone, Snapdragon 7s Gen 2, 8GB RAM, 256GB",
                "price": 4299000,
                "stock_quantity": 15,
                "category": "smartphone",
                "tags": "smartphone,xiaomi,redmi,midrange",
                "sku": "XIA-RN13P-256"
            },
            {
                "supplier": suppliers[4],
                "name": "Keyboard Razer BlackWidow V3",
                "description": "Full-size mechanical keyboard, Razer Green switches, RGB",
                "price": 1550000,
                "stock_quantity": 8,
                "category": "keyboard",
                "tags": "keyboard,razer,mechanical,rgb",
                "sku": "RAZ-BW-V3"
            }
        ]

        for data in products_data:
            supplier = data.pop("supplier")
            product = Product(supplier_id=supplier.id, **data)
            db.add(product)

        db.commit()
        print(f"[Seed] Created {len(products_data)} products")

        # Seed Users
        print("[Seed] Creating users...")

        users_data = [
            {
                "name": "John Doe",
                "email": "john.doe@email.com",
                "phone": "+62812111222",
                "latitude": -6.2088,
                "longitude": 106.8456,
                "city": "Jakarta",
                "province": "DKI Jakarta",
                "ai_access_enabled": True  # Premium user
            },
            {
                "name": "Jane Smith",
                "email": "jane.smith@email.com",
                "phone": "+62856333444",
                "latitude": -6.9175,
                "longitude": 107.6191,
                "city": "Bandung",
                "province": "Jawa Barat",
                "ai_access_enabled": True  # Premium user
            },
            {
                "name": "Bob Wilson",
                "email": "bob.wilson@email.com",
                "phone": "+62877555666",
                "latitude": -7.2575,
                "longitude": 112.7521,
                "city": "Surabaya",
                "province": "Jawa Timur",
                "ai_access_enabled": False  # Regular user
            }
        ]

        for data in users_data:
            user = User(**data)
            db.add(user)

        db.commit()
        print(f"[Seed] Created {len(users_data)} users")

        print("\n[Seed] ✅ Database seeding completed successfully!")
        print("\n[Seed] Summary:")
        print(f"  - Suppliers: {len(suppliers)}")
        print(f"  - Products: {len(products_data)}")
        print(f"  - Users: {len(users_data)}")
        print("\n[Seed] Sample data:")
        print("  User with AI access:")
        print(f"    - Email: john.doe@email.com (user_id: 1, Jakarta)")
        print(f"    - Email: jane.smith@email.com (user_id: 2, Bandung)")
        print("\n  Try AI search:")
        print('    curl -X POST http://localhost:3000/api/chat/search \\')
        print('      -H "Content-Type: application/json" \\')
        print('      -d \'{"user_id": 1, "query": "laptop gaming Jakarta budget 12 juta"}\'')

    except Exception as e:
        print(f"[Seed] ❌ Error during seeding: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()

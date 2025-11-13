#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Database models for AI Marketplace Platform

Tables:
- Suppliers: Business entities that list products
- Products: Items for sale with stock and pricing
- Users: Platform users with location and AI access
- Conversations: Chat sessions for tracking
- Messages: Individual messages in conversations
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, Index
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Supplier(Base):
    __tablename__ = "suppliers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    business_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255))  # For authentication
    phone = Column(String(50))
    address = Column(Text)
    latitude = Column(Float, nullable=False)  # For location-based search
    longitude = Column(Float, nullable=False)
    city = Column(String(100), index=True)
    province = Column(String(100))
    is_active = Column(Boolean, default=True)
    registration_date = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    products = relationship("Product", back_populates="supplier", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_supplier_location', 'latitude', 'longitude'),
    )


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    supplier_id = Column(Integer, ForeignKey("suppliers.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    price = Column(Float, nullable=False)
    stock_quantity = Column(Integer, nullable=False, default=0)
    category = Column(String(100), index=True)
    tags = Column(Text)  # Comma-separated tags for search
    sku = Column(String(100), unique=True, index=True)
    is_available = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    supplier = relationship("Supplier", back_populates="products")

    __table_args__ = (
        Index('idx_product_search', 'name', 'category'),
        Index('idx_product_price', 'price'),
    )


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255))  # For authentication
    phone = Column(String(50))
    latitude = Column(Float)
    longitude = Column(Float)
    city = Column(String(100))
    province = Column(String(100))
    role = Column(String(20), default="user")  # user, supplier, admin
    ai_access_enabled = Column(Boolean, default=False)  # Premium feature flag
    preferences = Column(Text)  # JSON string for user preferences
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_user_location', 'latitude', 'longitude'),
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    title = Column(String(255))  # Auto-generated from first message
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (
        Index('idx_message_conversation', 'conversation_id', 'timestamp'),
    )

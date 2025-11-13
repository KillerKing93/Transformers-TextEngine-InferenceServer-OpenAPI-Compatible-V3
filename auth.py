#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Authentication and authorization for AI Marketplace Platform

Features:
- JWT token generation and validation
- Password hashing with bcrypt
- Role-based access control (user, supplier, admin)
- Token refresh mechanism
- FastAPI dependencies for protected endpoints

Usage:
    from auth import get_current_user, create_access_token

    @app.post("/protected")
    def protected_route(current_user: dict = Depends(get_current_user)):
        return {"user": current_user}
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()


# Pydantic models
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserRegisterAuth(BaseModel):
    email: EmailStr
    password: str
    name: str
    role: str = "user"  # user, supplier, admin


# Password utilities
def hash_password(password: str) -> str:
    """Hash a password for storing."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a stored password against one provided by user."""
    return pwd_context.verify(plain_password, hashed_password)


# Token utilities
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.

    Args:
        data: Dictionary with user data (email, role, etc.)
        expires_delta: Optional token expiration time

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """
    Create JWT refresh token with longer expiration.

    Args:
        data: Dictionary with user data

    Returns:
        Encoded JWT refresh token string
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode JWT token.

    Args:
        token: JWT token string

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("email")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# FastAPI dependencies
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user.

    Usage:
        @app.get("/protected")
        def protected_route(current_user: dict = Depends(get_current_user)):
            return {"user": current_user}

    Returns:
        User data from token payload

    Raises:
        HTTPException: If token is invalid
    """
    token = credentials.credentials
    payload = verify_token(token)
    return payload


async def get_current_active_user(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current active user (additional checks can be added here).

    Args:
        current_user: User from token

    Returns:
        User data if active

    Raises:
        HTTPException: If user is inactive
    """
    # Add additional checks here (e.g., is_active flag from database)
    return current_user


async def require_role(required_role: str):
    """
    Dependency factory for role-based access control.

    Usage:
        @app.post("/admin/users", dependencies=[Depends(require_role("admin"))])
        def admin_only_route():
            return {"message": "Admin access"}

    Args:
        required_role: Required role (user, supplier, admin)

    Returns:
        Dependency function
    """
    async def role_checker(current_user: dict = Depends(get_current_user)):
        user_role = current_user.get("role", "user")
        if user_role != required_role and user_role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        return current_user

    return role_checker


# Helper for protected endpoints
def require_admin(current_user: dict = Depends(get_current_user)):
    """Require admin role for endpoint."""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def require_supplier(current_user: dict = Depends(get_current_user)):
    """Require supplier role for endpoint."""
    user_role = current_user.get("role")
    if user_role not in ["supplier", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Supplier access required"
        )
    return current_user


# Utility functions for user authentication
def authenticate_user(email: str, password: str, db_user: Dict[str, Any]) -> bool:
    """
    Authenticate user with email and password.

    Args:
        email: User email
        password: Plain password
        db_user: User data from database (must include 'hashed_password' key)

    Returns:
        True if authentication successful, False otherwise
    """
    if not db_user:
        return False
    if not verify_password(password, db_user.get("hashed_password", "")):
        return False
    return True


def create_tokens_for_user(email: str, role: str = "user", **extra_data) -> Token:
    """
    Create access and refresh tokens for user.

    Args:
        email: User email
        role: User role (user, supplier, admin)
        **extra_data: Additional data to include in token

    Returns:
        Token object with access_token, refresh_token, and token_type
    """
    token_data = {"email": email, "role": role, **extra_data}

    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )


# Example: Refresh token endpoint logic
def refresh_access_token(refresh_token: str) -> str:
    """
    Generate new access token from refresh token.

    Args:
        refresh_token: Valid refresh token

    Returns:
        New access token

    Raises:
        HTTPException: If refresh token is invalid or not a refresh type
    """
    payload = verify_token(refresh_token)

    # Check if it's a refresh token
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type"
        )

    # Create new access token
    token_data = {
        "email": payload.get("email"),
        "role": payload.get("role")
    }

    return create_access_token(token_data)

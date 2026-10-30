"""
auth.py - Authentication and User Management Module
Handles:
- User registration (Name, Phone, Email, Password, Confirm Password)
- Secure password hashing (SHA-256 with salt)
- Authentication / Sign In
- Local persistent storage (users.json)
"""

import os
import re
import json
import hashlib
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")

def _hash_password(password: str, salt: str = "scholarship_app_salt_2026") -> str:
    """Hashes a password with salt using SHA-256."""
    salted = f"{salt}_{password}_{salt}"
    return hashlib.sha256(salted.encode('utf-8')).hexdigest()

def init_auth_db():
    """Initializes users.json if not present and adds a sample test account."""
    if not os.path.exists(USERS_FILE):
        default_users = {
            "student@example.com": {
                "name": "Shamini K",
                "phone": "9876543210",
                "email": "student@example.com",
                "password_hash": _hash_password("Password@123"),
                "created_at": datetime.now().isoformat()
            }
        }
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_users, f, indent=2)

def load_users() -> Dict[str, Any]:
    """Loads users dictionary from disk."""
    init_auth_db()
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_users(users: Dict[str, Any]):
    """Saves users dictionary to disk."""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def validate_email(email: str) -> bool:
    """Checks for standard valid email pattern."""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email.strip()))

def validate_phone(phone: str) -> bool:
    """Checks for valid 10-digit phone number."""
    clean = re.sub(r'[\s\-+]', '', phone)
    return len(clean) >= 10 and clean.isdigit()

def register_user(
    name: str,
    phone: str,
    email: str,
    password: str,
    confirm_password: str
) -> Tuple[bool, str]:
    """
    Registers a new user with validation:
    - Name not empty
    - Valid phone number
    - Valid email format
    - Password at least 6 characters
    - Passwords match
    - Email not already taken
    """
    name = name.strip()
    phone = phone.strip()
    email = email.strip().lower()

    if not name:
        return False, "Please enter your full name."
    
    if not validate_phone(phone):
        return False, "Please enter a valid phone number (at least 10 digits)."

    if not validate_email(email):
        return False, "Please enter a valid email address (e.g., name@domain.com)."

    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    if password != confirm_password:
        return False, "Passwords do not match. Please verify your password."

    users = load_users()
    if email in users:
        return False, f"An account with email '{email}' already exists. Please Sign In."

    # Create user record
    users[email] = {
        "name": name,
        "phone": phone,
        "email": email,
        "password_hash": _hash_password(password),
        "created_at": datetime.now().isoformat()
    }
    save_users(users)
    return True, "Account created successfully! You can now Sign In."

def authenticate_user(email: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Authenticates user with email and password.
    Returns (success, user_dict, message).
    """
    email = email.strip().lower()
    users = load_users()

    if not email or not password:
        return False, None, "Please enter both Email and Password."

    if email not in users:
        return False, None, "No account found with this email. Please create an account."

    user_record = users[email]
    pwd_hash = _hash_password(password)

    if user_record.get("password_hash") != pwd_hash:
        return False, None, "Incorrect password. Please try again."

    # Return safe user profile (without password hash)
    safe_profile = {
        "name": user_record.get("name"),
        "email": user_record.get("email"),
        "phone": user_record.get("phone"),
        "created_at": user_record.get("created_at")
    }
    return True, safe_profile, "Login successful!"

"""Input validation utilities for security and data integrity."""
import re
from typing import Optional


def validate_email(email: str) -> bool:
    """Validate email format."""
    if not email or not isinstance(email, str):
        return False
    
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_student_id(student_id: str) -> bool:
    """Validate student ID format."""
    if not student_id or not isinstance(student_id, str):
        return False
    
    # Allow alphanumeric and hyphens, 3-20 characters
    pattern = r'^[a-zA-Z0-9-]{3,20}$'
    return bool(re.match(pattern, student_id))


def validate_username(username: str) -> bool:
    """Validate username format."""
    if not username or not isinstance(username, str):
        return False
    
    # Allow alphanumeric and underscore, 3-30 characters
    pattern = r'^[a-zA-Z0-9_]{3,30}$'
    return bool(re.match(pattern, username))


def validate_password(password: str) -> tuple[bool, Optional[str]]:
    """
    Validate password strength.
    
    Returns:
        (is_valid, error_message)
    """
    if not password or not isinstance(password, str):
        return False, "Password is required"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    
    if len(password) > 128:
        return False, "Password is too long (max 128 characters)"
    
    # Check for at least one letter and one number (basic strength)
    has_letter = bool(re.search(r'[a-zA-Z]', password))
    has_number = bool(re.search(r'[0-9]', password))
    
    if not (has_letter and has_number):
        return False, "Password must contain at least one letter and one number"
    
    return True, None


def validate_phone(phone: str) -> bool:
    """Validate phone number format."""
    if not phone or not isinstance(phone, str):
        return False
    
    # Allow digits, spaces, hyphens, parentheses, and plus sign
    # Length between 10-15 digits
    cleaned = re.sub(r'[\s\-\(\)\+]', '', phone)
    return bool(re.match(r'^\d{10,15}$', cleaned))


def validate_name(name: str) -> bool:
    """Validate person name."""
    if not name or not isinstance(name, str):
        return False
    
    # Allow letters, spaces, hyphens, apostrophes
    # Length between 2-100 characters
    if len(name) < 2 or len(name) > 100:
        return False
    
    pattern = r'^[a-zA-Z\s\-\'\.]+$'
    return bool(re.match(pattern, name))


def validate_department_code(code: str) -> bool:
    """Validate department code format."""
    if not code or not isinstance(code, str):
        return False
    
    # Allow uppercase letters and numbers, 2-10 characters
    pattern = r'^[A-Z0-9]{2,10}$'
    return bool(re.match(pattern, code.upper()))


def sanitize_string(text: str, max_length: int = 255) -> str:
    """
    Sanitize string input by removing potentially dangerous characters.
    
    Args:
        text: Input string
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove null bytes and control characters
    text = text.replace('\x00', '')
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Trim to max length
    return text[:max_length].strip()


def validate_zone(zone: str, allowed_zones: list[str]) -> bool:
    """Validate zone against allowed list."""
    if not zone or not isinstance(zone, str):
        return False
    
    return zone in allowed_zones


def validate_date_string(date_str: str) -> bool:
    """Validate date string format (YYYY-MM-DD)."""
    if not date_str or not isinstance(date_str, str):
        return False
    
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    if not re.match(pattern, date_str):
        return False
    
    # Additional validation: check if date is valid
    try:
        from datetime import datetime
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

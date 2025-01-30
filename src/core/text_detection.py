"""Text detection utilities using Tesseract OCR"""

import os
import cv2
import pytesseract
import re
from typing import Tuple, Optional, Union, List, Dict, Any
from .logging import app_logger
from .device import take_screenshot, get_screen_size
from .image_processing import _load_template, _take_and_load_screenshot, find_template, find_all_templates
from .config import CONFIG
from .debug import save_debug_region
import numpy as np
import json
from pathlib import Path

debug = True

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def get_text_regions(
    accept_location: Tuple[int, int], 
    device_id: str,
    existing_screenshot: Optional[np.ndarray] = None
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], Optional[np.ndarray]]:
    """Calculate regions for alliance tag and name extraction based on bracket location"""
    width, height = get_screen_size(device_id)
    
    if existing_screenshot is not None:
        img = existing_screenshot
    else:
        img = _take_and_load_screenshot(device_id)
        if img is None:
            return (0, 0, 0, 0), (0, 0, 0, 0), None
    
    app_logger.debug(f"Screen size: {width}x{height}")
    
    # Add debug logging for region calculations
    app_logger.debug(f"Accept button location: {accept_location}")
    
    # Calculate initial search region aligned with accept button
    x_offset = int(width * 0.48)
    y_offset = int(height * 0.015)  # 1.5% vertical search area
    
    # Get template size to ensure minimum search region
    template, template_config = _load_template('left_bracket')
    if template is not None:
        min_width = template.shape[1] * 3
        min_height = template.shape[0] * 3
    else:
        min_width = int(width * 0.1)
        min_height = int(height * 0.03)
    
    # Search region centered on accept button's vertical position, shifted up slightly
    x1 = max(0, accept_location[0] - x_offset)
    y1 = max(0, accept_location[1] - y_offset - int(height * 0.02))  # Shift up by 2%
    y2 = min(height, y1 + int(height * 0.025))  # Fixed height of 2.5%
    x2 = accept_location[0] - int(width * 0.02)
    
    # Ensure minimum region size
    if x2 - x1 < min_width:
        x2 = min(width, x1 + min_width)
    if y2 - y1 < min_height:
        y2 = min(height, y1 + min_height)
    
    # Take screenshot and crop to search region
    if not take_screenshot(device_id):
        return (0, 0, 0, 0), (0, 0, 0, 0), None
        
    img = cv2.imread('tmp/screen.png')
    if img is None:
        return (0, 0, 0, 0), (0, 0, 0, 0), None
        
    # Crop image to search region
    search_region = img[y1:y2, x1:x2]
    
    # Find brackets within cropped region
    left_brackets = find_all_templates(
        device_id,
        "left_bracket",
        search_region=(x1, y1, x2, y2)
    )
    
    right_brackets = find_all_templates(
        device_id,
        "right_bracket",
        search_region=(x1, y1, x2, y2)
    )
    
    # Filter brackets by vertical alignment with accept button
    valid_left = []
    valid_right = []
    
    for left in left_brackets:
        if abs(left[1] - accept_location[1]) <= int(height * 0.02):  # Must be within 2% of accept button
            for right in right_brackets:
                if abs(right[1] - left[1]) <= 8:  # Brackets must align with each other
                    x_distance = right[0] - left[0]
                    if 25 <= x_distance <= 200:
                        valid_left.append(left)
                        valid_right.append(right)
                        break
    
    if valid_left and valid_right:
        left_bracket = min(valid_left, key=lambda x: x[0])
        right_bracket = min(valid_right, key=lambda x: x[0])
        app_logger.debug(f"Selected brackets - Left: {left_bracket}, Right: {right_bracket}")
        
        # Get bracket width
        template, _ = _load_template('left_bracket')
        bracket_width = template.shape[1] if template is not None else int(width * 0.01)
        
        # Calculate vertical bounds based on bracket position
        y_center = left_bracket[1]
        y_padding = int(height * 0.02)
        
        # Now use bracket_width in region calculations
        alliance_region = (
            left_bracket[0] + bracket_width - 15,
            max(0, y_center - y_padding),
            right_bracket[0] + 15,
            min(height, y_center + y_padding)
        )
        app_logger.debug(f"Calculated alliance region: {alliance_region}")
        
        name_region = (
            right_bracket[0] + bracket_width,
            max(0, y_center - y_padding),
            x2,
            min(height, y_center + y_padding)
        )
        
        save_debug_region(device_id, alliance_region, "alliance")
        return alliance_region, name_region, img
    
    # Fallback case with wider ratio
    app_logger.warning("No valid bracket pairs found, using fallback ratio")
    split_x = x1 + int(x_offset * 0.5)
    y_padding = int(height * 0.015)  # Add padding for better text capture
    
    # Move the region down closer to accept button
    y_center = accept_location[1] - int(height * 0.015)  # Reduced from 0.03 to 0.015
    
    alliance_region = (
        x1 + int(width * 0.02),  # Add slight padding from left
        max(0, y_center - y_padding),  # Center around adjusted y position
        split_x,
        min(height, y_center + y_padding)
    )
    name_region = (split_x, y1, x2, y2)
    
    # Save debug image for fallback case too
    save_debug_region(device_id, alliance_region, "alliance")
    
    return alliance_region, name_region, img

def clean_text(text: str) -> str:
    """Strip non-alphanumeric characters from text"""
    return re.sub(r'[^a-zA-Z0-9]', '', text)

def extract_text_from_region(device_id: str, region: Tuple[int, int, int, int], languages: Union[str, List[str]] = 'eng', img: Optional[np.ndarray] = None) -> str:
    if img is None:
        if not take_screenshot(device_id):
            return "", ""
        img = cv2.imread('tmp/screen.png')
        if img is None:
            return "", ""
    
    x1, y1, x2, y2 = region
    cropped = img[y1:y2, x1:x2]
    
    if languages == 'eng':
        # Convert to grayscale
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        # Higher scale factor for better detail
        scale = 8
        enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Simple binary threshold
        _, binary = cv2.threshold(enlarged, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Optional: Add slight dilation to connect components
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        # Save debug images
        cv2.imwrite('tmp/debug_alliance_original.png', cropped)
        cv2.imwrite('tmp/debug_alliance_processed.png', binary)
        
        # OCR with specific config for pixel font
        config = (
            '--psm 7 '  # Single line mode
            '--oem 1 '  # LSTM only
            f'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890[] '
            '-c tessedit_write_images=1 '
            '-c textord_min_linesize=2 '
            '-c edges_max_children_per_outline=40'
        )
        text = pytesseract.image_to_string(binary, lang=languages, config=config).strip()
        original_text = text
        
        # Clean up text but preserve case
        text = text.replace('—', '').replace('–', '').strip()
        
        # Try to extract text between brackets first
        bracket_match = re.search(r'[\[|\(](.*?)[\]|\)]', text)
        if bracket_match:
            return bracket_match.group(1), original_text
            
        # If no brackets, look for 3-4 letter sequences that match alliance patterns
        words = re.findall(r'[A-Za-z0-9]{3,4}', text)
        if words:
            # Take first word that matches length of known alliances
            for word in words:
                if len(word) in {3, 4}:  # Most alliance tags are 3-4 chars
                    return word, original_text
        
        return "", original_text
        
    return "", original_text

def log_rejected_alliance(alliance_text: str, original_text: str = ""):
    """Log rejected alliance names to a file and store debug images"""
    from datetime import datetime
    import shutil
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create reject directory with timestamp
        reject_dir = f'tmp/rejects/{timestamp}'
        os.makedirs(reject_dir, exist_ok=True)
        
        # Log text info with original parsed text
        with open('logs/rejected_alliances.log', 'a', encoding='utf-8') as f:
            f.write(f"{timestamp} - Final: {alliance_text}\n")
            f.write(f"    Original OCR text: {original_text}\n")
            f.write(f"    Debug files: {reject_dir}/\n\n")
            
        # Copy debug images if they exist
        debug_files = {
            'processed': 'tmp/debug_alliance_processed.png',
            'original': 'tmp/debug_alliance_original.png',
            'region': 'tmp/debug_alliance.png',
            'full': 'tmp/debug_alliance_full.png',
            'screen': 'tmp/screen.png'
        }
        
        for img_type, src_path in debug_files.items():
            if os.path.exists(src_path):
                dst_path = f'{reject_dir}/{img_type}.png'
                shutil.copy2(src_path, dst_path)
                app_logger.debug(f"Saved {img_type} image to {dst_path}")
                
    except Exception as e:
        app_logger.error(f"Failed to log rejected alliance: {e}")

def build_alliance_whitelist() -> str:
    """Build whitelist of characters from alliance tags in control list"""
    # Get alliance tags from main config
    alliance_tags = CONFIG.get('control_list', {}).get('whitelist', {}).get('alliance', [])
    
    # Create set of unique characters exactly as they appear
    chars = set()
    for tag in alliance_tags:
        chars.update(tag)
    
    # Convert to sorted string
    whitelist = ''.join(sorted(chars))
    app_logger.debug(f"Alliance whitelist: {whitelist}")
    return whitelist

# Build alliance whitelist from config
ALLIANCE_CHARS = build_alliance_whitelist()

# Export CONFIG's control list for compatibility
CONTROL_LIST = CONFIG.get('control_list', {
    "whitelist": {"alliance": []}, 
    "blacklist": {"alliance": []}
})

__all__ = ['CONTROL_LIST', 'ALLIANCE_CHARS', 'extract_text_from_region', 
           'get_text_regions', 'log_rejected_alliance'] 
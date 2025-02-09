"""Image processing utilities"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple
from .logging import app_logger
from .device import take_screenshot
from .config import CONFIG
import os

def _load_template(template_name: str) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """Load template and its config"""
    template_config = CONFIG['templates'].get(template_name)
    if not template_config:
        app_logger.error(f"Template {template_name} not found in config")
        return None, None
        
    template = cv2.imread(f"config/{template_config['path']}")
    if template is None:
        app_logger.error(f"Failed to load template: config/{template_config['path']}")
        return None, None
        
    return template, template_config

def _take_and_load_screenshot(device_id: str) -> Optional[np.ndarray]:
    """Take and load a screenshot"""
    if not take_screenshot(device_id):
        app_logger.error("Failed to take screenshot")
        return None
        
    img = cv2.imread('tmp/screen.png')
    if img is None:
        app_logger.error("Failed to load screenshot")
        return None
        
    return img

def find_template(
    device_id: str,
    template_name: str
) -> Optional[Tuple[int, int]]:
    """Find template in image and return center coordinates"""
    try:
        app_logger.debug(f"Looking for template: {template_name}")
        
        template, template_config = _load_template(template_name)
        if template is None:
            app_logger.debug(f"Failed to load template: {template_name}")
            return None
            
        app_logger.debug(f"Template loaded successfully. Shape: {template.shape}")
        
        # Take screenshot first
        if not take_screenshot(device_id):
            app_logger.error("Failed to take screenshot")
            return None
            
        img = cv2.imread('tmp/screen.png')
        if img is None:
            app_logger.debug("Failed to load screenshot")
            return None
            
        app_logger.debug(f"Screenshot loaded successfully. Shape: {img.shape}")
        
        # Match template
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Get threshold from template config or use default
        threshold = template_config.get('threshold', CONFIG['match_threshold'])
        app_logger.debug(f"Match values - Max: {max_val:.4f}, Min: {min_val:.4f}, Threshold: {threshold}")
        app_logger.debug(f"Match location - Max: {max_loc}, Min: {min_loc}")
        
        # Fix the threshold comparison
        if max_val < threshold:  # Remove the incorrect "threshold - -0.16"
            app_logger.debug(f"Match value {max_val:.4f} below threshold {threshold}")
            return None
            
        app_logger.debug(f"Match value {max_val:.4f} EXCEEDS threshold {threshold} !")
        
        # Save debug image
        debug_img = img.copy()
        h, w = template.shape[:2]
        cv2.rectangle(debug_img, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)
        cv2.putText(debug_img, f"{max_val:.3f}", (max_loc[0], max_loc[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(f'tmp/debug_find_{template_name}.png', debug_img)
        
        # Get template dimensions and calculate center point
        center_x = max_loc[0] + w//2
        center_y = max_loc[1] + h//2
        
        app_logger.debug(f"Found match at center point: ({center_x}, {center_y})")
        return (center_x, center_y)
        
    except Exception as e:
        app_logger.error(f"Error finding template {template_name}: {e}")
        return None
    
def find_all_templates(
    device_id: str,
    template_name: str,
    search_region: Tuple[int, int, int, int] = None
) -> list[Tuple[int, int]]:
    """Find all template matches in image and return center coordinates"""
    try:
        template, template_config = _load_template(template_name)
        if template is None:
            return []
            
        h, w = template.shape[:2]
        
        # if template_name == "accept":
        #     img = cv2.imread('2.png')
        # else:
        #     img = _take_and_load_screenshot(device_id)
        img = _take_and_load_screenshot(device_id)
        if img is None:
            return []
            
        # Get region to search
        if search_region:
            x1, y1, x2, y2 = search_region
            img_region = img[y1:y2, x1:x2]
        else:
            img_region = img
            
        result = cv2.matchTemplate(img_region, template, cv2.TM_CCOEFF_NORMED)
        threshold = template_config.get('threshold', CONFIG['match_threshold'])
            
        matches = []
        result_copy = result.copy()
        
        while True:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_copy)
            if max_val < threshold:
                break
                
            # Store match with confidence
            center_x = max_loc[0] + w//2
            center_y = max_loc[1] + h//2
            matches.append((center_x, center_y, max_val))
            
            # Suppress region
            x1_sup = max(0, max_loc[0] - w//2)
            y1_sup = max(0, max_loc[1] - h//2)
            x2_sup = min(result_copy.shape[1], max_loc[0] + w//2)
            y2_sup = min(result_copy.shape[0], max_loc[1] + h//2)
            result_copy[y1_sup:y2_sup, x1_sup:x2_sup] = 0
        
        # Adjust coordinates if search region was used
        adjusted_matches = []
        for x, y, conf in matches:
            if search_region:
                x += search_region[0]
                y += search_region[1]
            adjusted_matches.append((x, y))
            
        # Save debug image
        _save_debug_image(img, template_config['path'], matches, search_region, (w, h))
        
        app_logger.debug(f"Found {len(matches)} matches for {template_name} with threshold {threshold}")
        return adjusted_matches
        
    except Exception as e:
        app_logger.error(f"Error finding templates: {e}")
        return []
    
def wait_for_image(
    device_id: str,
    template_name: str,
    timeout: float = 120.0,
    interval: float = 1.0
) -> Optional[Tuple[int, int]]:
    """Wait for template to appear in screenshot"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        coords = find_template(device_id, template_name)
        if coords:
            return coords
        time.sleep(interval)
    return None

def compare_screenshots(img1: np.ndarray, img2: np.ndarray) -> bool:
    """
    Compare two screenshots to detect if they are nearly identical
    Returns True if images are the same, False if different
    """
    if img1 is None or img2 is None:
        return False
        
    if img1.shape != img2.shape:
        return False
        
    # Calculate structural similarity index
    score = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)[0][0]
    
    # Use match_threshold from config for consistency with template matching
    return score >= CONFIG['match_threshold']

def _save_debug_image(
    img: np.ndarray, 
    template_name: str,
    matches: list[Tuple[int, int, float]] = None,
    search_region: Tuple[int, int, int, int] = None,
    template_size: Tuple[int, int] = None
) -> None:
    """Save debug image with matches and search region highlighted"""
    try:
        debug_img = img.copy()
        
        # Draw search region if provided
        if search_region:
            x1, y1, x2, y2 = search_region
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
        # Draw matches if provided
        if matches and template_size:
            w, h = template_size
            for x, y, conf in matches:
                rect_x = x - w//2
                rect_y = y - h//2
                cv2.rectangle(debug_img, (rect_x, rect_y), 
                            (rect_x + w, rect_y + h), (0, 255, 0), 2)
                cv2.putText(debug_img, f"{conf:.3f}", (rect_x, rect_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
        # Save debug image
        template_name = os.path.basename(template_name)
        cv2.imwrite(f'tmp/debug_template_{template_name}.png', debug_img)
        
    except Exception as e:
        app_logger.error(f"Error saving debug image: {e}")

def find_and_tap_template(
    device_id: str, 
    template_name: str,
    error_msg: Optional[str] = None,
    success_msg: Optional[str] = None,
    long_press: bool = False,
    press_duration: float = 1.0,
    critical: bool = False,
    timeout: float = None
) -> bool:
    """Find and tap a template on screen"""
    if timeout:
        location = wait_for_image(device_id, template_name, timeout=timeout)
    else:
        location = find_template(device_id, template_name)
    
    if location is None:
        if error_msg:
            if critical:
                app_logger.error(error_msg)
            else:
                app_logger.info(error_msg)
        return False
        
    if success_msg:
        app_logger.info(success_msg)
        
    # Import here to avoid circular dependency
    from src.game.controls import humanized_tap, humanized_long_press
        
    if long_press:
        humanized_long_press(device_id, location[0], location[1], duration=press_duration)
    else:
        humanized_tap(device_id, location[0], location[1])
        
    return True

def find_match_image(
    template_file_name: str,
    img,
    search_region: Tuple[int, int, int, int] = None
) -> list[Tuple[int, int]]:
    """Find all template matches in image and return center coordinates"""
    try:
        template = cv2.imread(f"config/{template_file_name}")
        h, w = template.shape[:2]
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        matches = []
        result_copy = result.copy()
        while True:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_copy)
            if max_val < threshold:
                break
                
            # Store match with confidence
            center_x = max_loc[0] + w//2
            center_y = max_loc[1] + h//2
            matches.append((center_x, center_y, max_val))
            
            # Suppress region
            x1_sup = max(0, max_loc[0] - w//2)
            y1_sup = max(0, max_loc[1] - h//2)
            x2_sup = min(result_copy.shape[1], max_loc[0] + w//2)
            y2_sup = min(result_copy.shape[0], max_loc[1] + h//2)
            result_copy[y1_sup:y2_sup, x1_sup:x2_sup] = 0
        
        # Adjust coordinates if search region was used
        adjusted_matches = []
        for x, y, conf in matches:
            adjusted_matches.append((x, y))
            
        # Save debug image
        _save_debug_image(img, template_file_name, matches, search_region, (w, h))
        
        app_logger.debug(f"Found {len(matches)} matches for {template_file_name} with threshold {threshold}")
        return adjusted_matches
        
    except Exception as e:
        app_logger.error(f"Error finding templates: {e}")
        return []
    

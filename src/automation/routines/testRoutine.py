from typing import Tuple
import os
import cv2
import re
import numpy as np
import pytesseract
from src.automation.routines.routineBase import TimeCheckRoutine
from src.core.logging import app_logger
from src.core.config import CONFIG
from src.core.network_sniffing import start_network_capture
from src.core.device import take_screenshot
from src.core.adb import get_screen_size, press_back
from src.game.controls import human_delay, humanized_tap, handle_swipes
from typing import Optional, Tuple
from src.core.audio import play_beep
import numpy as np
import shutil
import subprocess
from pathlib import Path
import time
from src.core.text_detection import (
    extract_text_from_region, 
    get_text_regions, 
    log_rejected_alliance,
    CONTROL_LIST
)

class testRoutine(TimeCheckRoutine):
    force_home: bool = True

    def __init__(self, device_id: str, interval: int, last_run: float = None, automation=None):
        super().__init__(device_id, interval, last_run, automation)
        self.secretary_types = ["strategy", "security", "development", "science", "interior"]
        self.capture = None
        self.manual_deny = False

    def _execute(self) -> bool:
        """Start secretary automation sequence"""
        return self.execute_with_error_handling(self._execute_internal)
    
    def _execute_internal(self) -> bool:
        """Internal execution logic"""
        self.automation.game_state["is_home"] = False
        self.open_profile_menu(self.device_id)
        self.open_secretary_menu(self.device_id)
        return self.process_all_secretary_positions()

    def sortTest(self):
        accept_locations = self.find_accept_buttons()
        for location in accept_locations:
            humanized_tap(self.device_id, location[0], location[1])            
            break
        # find_and_tap_template(
        #     self.device_id,
        #     "accept",
        #     error_msg=f"Failed to accept candidate for temp",
        #     success_msg=f"Accepting candidate for temp"
        # )
        return True

    def process_secretary_position(self, name: str) -> bool:
        """Process a single secretary position"""
        try:
            # Find and click secretary position
            if not find_and_tap_template(
                self.device_id,
                name,
                error_msg=f"Could not find {name} secretary position",
                critical=True
            ):
                return True  # Continue with next position
            
            human_delay(CONFIG['timings']['tap_delay'])
            
            # Find and click list button
            if not find_and_tap_template(
                self.device_id,
                "list",
                error_msg="List button not found",
                critical=True,
                timeout=CONFIG['timings']['list_timeout']
            ):
                return False

            accept_locations = self.find_accept_buttons()
            if accept_locations:
                # Scroll to top if needed
                if len(accept_locations) > 5:
                    handle_swipes(self.device_id, direction="up")
                    human_delay(CONFIG['timings']['settle_time'] * 2)
                    accept_locations = self.find_accept_buttons()
                
                processed = 0
                accepted = 0

                max_accept_count = CONFIG["max_accept_count"]
                while processed < max_accept_count:  # Max 8 applicants
                    # if not take_screenshot(self.device_id):
                    #     break
                        
                    current_screenshot = cv2.imread('tmp/screen.png')
                    if current_screenshot is None:
                        break
                    
                    accept_locations = self.find_accept_buttons()
                    if not accept_locations:
                        break
                    
                    topmost_accept = accept_locations[0]
                    alliance_region, name_region, screenshot = get_text_regions(
                        topmost_accept, 
                        self.device_id,
                        existing_screenshot=current_screenshot
                    )
                    
                    if screenshot is None:
                        continue

                    alliance_text, original_text = extract_text_from_region(
                        self.device_id, 
                        alliance_region, 
                        languages='eng', 
                        img=screenshot
                    )
                    app_logger.info(f"##### accept alliance:{alliance_text},{original_text}")

                    # No whitelist - accept all
                    for location in accept_locations:
                        humanized_tap(self.device_id, location[0], location[1])            
                        break
                    
                    processed += 1
                    human_delay(CONFIG['timings']['settle_time'])
                                
            # Exit menus with verification
            if not self.exit_to_secretary_menu():
                app_logger.error("Failed to exit to secretary menu")
                return False
            
            return True
            
        except Exception as e:
            app_logger.error(f"Error processing secretary position: {e}")
            if not self.exit_to_secretary_menu():
                app_logger.error("Failed to exit to secretary menu after error")
                return False
            return True

    def process_all_secretary_positions(self) -> bool:
        """Process all secretary positions that have applicants"""
        positions_to_process = self.find_positions_with_applicants()
        
        if not positions_to_process:
            app_logger.info("No positions with applicants found")
            return True
        
        for name in positions_to_process:
            if not self.process_secretary_position(name):
                return False
        return True

    def find_positions_with_applicants(self) -> list[str]:
        """Find all secretary positions that have applicants"""
        try:
            positions_to_process = []
            
            # Find all secretary positions
            all_positions = {}
            for position_type in self.secretary_types:
                positions = find_all_templates(
                    self.device_id,
                    position_type
                )
                if positions:
                    all_positions[position_type] = positions[0]  # Take first match for each type
                    app_logger.debug(f"Found {position_type} position at ({positions[0][0]}, {positions[0][1]})")
            # Find all applicant icons
            applicant_locations = find_all_templates(
                self.device_id,
                "has_applicant"
            )
            
            if not applicant_locations:
                app_logger.debug("No applicant icons found")
                return []
            
            app_logger.debug(f"Found {len(applicant_locations)} applicant icons:")
            for i, (x, y) in enumerate(applicant_locations):
                app_logger.debug(f"  Applicant {i+1}: ({x}, {y})")
            
            # For each position, check if there's an applicant icon nearby
            for position_type, pos_loc in all_positions.items():
                pos_x, pos_y = pos_loc
                
                # Check each applicant icon
                for app_x, app_y in applicant_locations:
                    x_diff = app_x - pos_x
                    y_diff = app_y - pos_y
                    # Check if applicant icon is within 100 pixels horizontally and 25 pixels vertically
                    if abs(x_diff) <= 100 and abs(y_diff) <= 28:
                        positions_to_process.append(position_type)
                        app_logger.info(f"Found applicant for {position_type} position")
                        break
            
            return positions_to_process
            
        except Exception as e:
            app_logger.error(f"Error finding positions with applicants: {e}")
            return []

    def find_accept_buttons(self) -> list[Tuple[int, int]]:
        """Find all accept buttons on the screen and sort by Y coordinate"""
        try:
            matches = find_all_templates(
                self.device_id,
                "accept"
            )
            if not matches:
                return []
            
            # Sort by Y coordinate (ascending) and X coordinate (ascending) for same Y
            sorted_matches = sorted(matches, key=lambda x: (x[1], x[0]))
            if sorted_matches:
                app_logger.debug(f"Found {len(sorted_matches)} accept buttons")
                app_logger.debug(f"Topmost button at coordinates: ({sorted_matches[0][0]}, {sorted_matches[0][1]})")
            return sorted_matches
        
        except Exception as e:
            app_logger.error(f"Error finding accept buttons: {e}")
            return []

    def open_profile_menu(self, device_id: str) -> bool:
        """Open the profile menu"""
        try:
            width, height = get_screen_size(device_id)
            profile = CONFIG['ui_elements']['profile']
            profile_x = int(width * float(profile['x'].strip('%')) / 100)
            profile_y = int(height * float(profile['y'].strip('%')) / 100)
            humanized_tap(device_id, profile_x, profile_y)

            # Look for notification indicators
            notification = wait_for_image(
                device_id,
                "awesome",
                timeout=CONFIG['timings']['menu_animation'],
            )
            
            if notification:
                humanized_tap(device_id, notification[0], notification[1])
                press_back(device_id)
                human_delay(CONFIG['timings']['menu_animation'])

            return True
        except Exception as e:
            app_logger.error(f"Error opening profile menu: {e}")
            return False

    def open_secretary_menu(self, device_id: str) -> bool:
        """Open the secretary menu"""
        try:
            width, height = get_screen_size(device_id)
            secretary = CONFIG['ui_elements']['secretary_menu']

            # Click secretary menu with randomization
            secretary_x = int(width * float(secretary['x'].strip('%')) / 100)
            secretary_y = int(height * float(secretary['y'].strip('%')) / 100)
            humanized_tap(device_id, secretary_x, secretary_y)
        
            return True
        except Exception as e:
            app_logger.error(f"Error opening secretary menu: {e}")
            return False

    def exit_to_secretary_menu(self) -> bool:
        """Exit back to secretary menu"""
        try:
            max_attempts = 10
            for _ in range(max_attempts):
                if self.verify_secretary_menu():
                    return True
                    
                press_back(self.device_id)
                human_delay(CONFIG['timings']['menu_animation'])
                
            app_logger.error("Failed to return to secretary menu")
            return False
            
        except Exception as e:
            app_logger.error(f"Error exiting to secretary menu: {e}")
            return False
    
    def verify_secretary_menu(self) -> bool:
        """Verify we're in the secretary menu"""
        return wait_for_image(
            self.device_id,
            "president",
            timeout=CONFIG['timings']['menu_animation']
        ) is not None

def _take_and_load_screenshot(device_id: str, template_name: str) -> Optional[np.ndarray]:
    """Take and load a screenshot"""
    if not take_screenshot(device_id, template_name):
        app_logger.error("Failed to take screenshot")
        return None
        
    img = cv2.imread('tmp/screen.png')
    if img is None:
        app_logger.error("Failed to load screenshot")
        return None
        
    return img

def take_screenshot(device_id: str, template_name: str) -> bool:
    """Take screenshot and pull to local tmp directory"""
    try:
        ensure_dir("tmp")
        if template_name == "has_applicant":
            new_path = shutil.copy('1.png', 'tmp/screen.png')
            return True
        if template_name == "accept":
            new_path = shutil.copy('2.png', 'tmp/screen.png')
            return True
        
        # Take screenshot on device
        cmd = f"adb -s {device_id} shell screencap -p /sdcard/screen.png"
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            app_logger.error(f"Failed to take screenshot: {result.stderr}")
            return False
            
        # Pull screenshot to local tmp directory
        cmd = f"adb -s {device_id} pull /sdcard/screen.png tmp/screen.png"
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            app_logger.error(f"Failed to pull screenshot: {result.stderr}")
            return False
            
        # Clean up device screenshot
        cleanup_device_screenshots(device_id)
        return True
        
    except Exception as e:
        app_logger.error(f"Error taking screenshot: {e}")
        return False

def load_template(template_name: str) -> Tuple[Optional[np.ndarray], Optional[dict]]:
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

def find_all_templates(
    device_id: str,
    template_name: str,
    search_region: Tuple[int, int, int, int] = None
) -> list[Tuple[int, int]]:
    """Find all template matches in image and return center coordinates"""
    try:
        template, template_config = load_template(template_name)
        if template is None:
            return []
            
        h, w = template.shape[:2]
        
        img = _take_and_load_screenshot(device_id, template_name)
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

def ensure_dir(path: str) -> None:
    """Ensure directory exists"""
    Path(path).mkdir(exist_ok=True)

def cleanup_device_screenshots(device_id: str) -> None:
    """Clean up screenshots from device"""
    try:
        cmd = f"adb -s {device_id} shell rm -f /sdcard/screen*.png"
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            app_logger.debug("Cleaned up device screenshots")
        else:
            app_logger.warning(f"Failed to clean device screenshots: {result.stderr}")
    except Exception as e:
        app_logger.error(f"Error cleaning device screenshots: {e}")

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

def find_template(
    device_id: str,
    template_name: str
) -> Optional[Tuple[int, int]]:
    """Find template in image and return center coordinates"""
    try:
        app_logger.debug(f"Looking for template: {template_name}")
        
        template, template_config = load_template(template_name)
        if template is None:
            app_logger.debug(f"Failed to load template: {template_name}")
            return None
            
        app_logger.debug(f"Template loaded successfully. Shape: {template.shape}")
        
        # Take screenshot first
        if not take_screenshot(device_id, template_name):
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
        cv2.rectangle(debug_img, max_loc, (max_loc[0] + w, max_loc[1] + h), (255, 255, 255), 2)
        cv2.putText(debug_img, f"{max_val:.3f}", (max_loc[0], max_loc[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(f'tmp/debug_find_{template_name}.png', debug_img)
        
        # Get template dimensions and calculate center point
        center_x = max_loc[0] + w//2
        center_y = max_loc[1] + h//2
        
        app_logger.debug(f"Found match at center point: ({center_x}, {center_y})")
        return (center_x, center_y)
        
    except Exception as e:
        app_logger.error(f"Error finding template {template_name}: {e}")
        return None

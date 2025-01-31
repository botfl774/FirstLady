"""Game control utilities for input and interaction"""

import time
import random
from src.core.image_processing import find_and_tap_template, find_template, wait_for_image
from src.core.logging import app_logger
from src.core.device import get_screen_size
from src.core.adb import force_stop_package, launch_package, press_back, swipe_screen, tap_screen, long_press_screen
from src.core.config import CONFIG

def human_delay(delay: float):
    """Add a human-like delay between actions"""
    time.sleep(delay * CONFIG.get('sleep_multiplier', 1.0))

def handle_swipes(device_id: str, direction: str = "up", num_swipes: int = 8) -> None:
    """Handle scrolling with swipes"""
    width, height = get_screen_size(device_id)
    
    swipe_cfg = CONFIG['ui_elements']['swipe']
    variance_pct = float(CONFIG['randomization']['swipe_variance']['position'].strip('%')) / 100
    
    for i in range(num_swipes):
        if direction == "up":
            start_x = int(width * float(swipe_cfg['start_x'].strip('%')) / 100)
            start_y = int(height * float(swipe_cfg['start_y'].strip('%')) / 100)
            end_x = int(width * float(swipe_cfg['end_x'].strip('%')) / 100)
            end_y = int(height * float(swipe_cfg['end_y'].strip('%')) / 100)
        else:  # down
            start_x = int(width * float(swipe_cfg['start_x'].strip('%')) / 100)
            start_y = int(height * float(swipe_cfg['end_y'].strip('%')) / 100)
            end_x = int(width * float(swipe_cfg['end_x'].strip('%')) / 100)
            end_y = int(height * float(swipe_cfg['start_y'].strip('%')) / 100)
        
        # Add slight variance to avoid detection
        start_x += int(width * random.uniform(-variance_pct, variance_pct))
        start_y += int(height * random.uniform(-variance_pct, variance_pct))
        end_x += int(width * random.uniform(-variance_pct, variance_pct))
        end_y += int(height * random.uniform(-variance_pct, variance_pct))
        
        duration = random.randint(
            CONFIG['timings']['swipe_duration']['min'],
            CONFIG['timings']['swipe_duration']['max']
        )
        
        swipe_screen(device_id, start_x, start_y, end_x, end_y, duration)
        human_delay(CONFIG['timings']['scroll_delay'])

def humanized_tap(device_id: str, x: int, y: int, critical: bool = False, delay: float = None) -> None:
    """Perform a humanized tap with randomization and delay
    
    Args:
        device_id: ADB device ID
        x: Target x coordinate
        y: Target y coordinate
        critical: Whether to use critical radius for randomization
        delay: Optional custom delay (uses tap_delay if None)
    """
    radius = CONFIG['randomization']['critical_radius'] if critical else CONFIG['randomization']['normal_radius']
    rand_x = x + random.randint(-radius, radius)
    rand_y = y + random.randint(-radius, radius)

    tap_screen(device_id, rand_x, rand_y)
    human_delay(delay or CONFIG['timings']['tap_delay'])

def humanized_long_press(device_id: str, x: int, y: int, duration: float = 1.0, critical: bool = False) -> bool:
    """Perform a humanized long press with randomization
    
    Args:
        device_id: Device identifier
        x: X coordinate
        y: Y coordinate
        duration: Press duration in seconds
        critical: Whether this is a critical press requiring higher precision
    """
    try:
        # Apply position randomization
        radius = CONFIG['randomization']['critical_radius'] if critical else CONFIG['randomization']['normal_radius']
        rand_x = x + random.randint(-radius, radius)
        rand_y = y + random.randint(-radius, radius)
        
        # Convert duration to milliseconds and add slight randomization
        ms_duration = int(duration * 1000 * random.uniform(0.9, 1.1))
        
        # Execute long press
        success = long_press_screen(device_id, rand_x, rand_y, ms_duration)
        if success:
            human_delay(CONFIG['timings']['tap_delay'])
            return True
        return False
        
    except Exception as e:
        app_logger.error(f"Error performing humanized long press: {e}")
        return False
 
def launch_game(device_id: str):
    """Launch the game"""
    force_stop_package(device_id, CONFIG['package_name'])
    time.sleep(CONFIG['timings']['app_close_wait'])
    launch_package(device_id, CONFIG['package_name'])
    
    start_time = time.time()
    while time.time() - start_time < CONFIG['timings']['launch_max_wait']:
        # Check for start button first
        # start_loc = find_template(device_id, "start")
        # if start_loc:
        #     app_logger.debug("Found start button, clicking it")
        #     humanized_tap(device_id, start_loc[0], start_loc[1])
        #     human_delay(CONFIG['timings']['menu_animation'])
        
        # Check for home icon
        home_loc = find_template(device_id, "home")
        if home_loc:
            app_logger.debug("Found home icon")
            time.sleep(CONFIG['timings']['launch_wait'])
            navigate_home(device_id, True)
            return True

        #small delay between checks    
        human_delay(5)
    
    app_logger.error("Could not find home icon after launch")
    return False

def navigate_home(device_id: str, force: bool = False) -> bool:
    """Navigate to home screen"""
    try:
        # Check if already at home
        if not force:
            home_loc = find_template(device_id, "home")
            if home_loc:
                app_logger.debug("Already at home screen")
                return True
            
        # Press back until we find home screen
        max_attempts = CONFIG.get('max_home_attempts', 10)
        for attempt in range(max_attempts):
            # Not found, press back and wait
            press_back(device_id)
            human_delay(CONFIG['timings']['menu_animation'])
            
            # Take screenshot and look for home
            quit_loc = find_template(device_id, "quit")
            if quit_loc:
                app_logger.debug("Found quit button")

                # Press back again to get to home
                while quit_loc:
                    press_back(device_id)
                    human_delay(CONFIG['timings']['menu_animation'])
                    quit_loc = find_template(device_id, "quit")
                    
                return True
                
            # Not found, press back and wait
            human_delay(CONFIG['timings']['menu_animation'])
            
        app_logger.error("Failed to find home screen after maximum attempts")
        return False
        
    except Exception as e:
        app_logger.error(f"Error navigating home: {e}")
        return False
   
from typing import Tuple
import cv2
import re
import numpy as np
import pytesseract
from src.automation.routines.routineBase import TimeCheckRoutine
from src.core.logging import app_logger
from src.core.config import CONFIG
from src.core.image_processing import _take_and_load_screenshot, find_template, find_all_templates, wait_for_image, compare_screenshots, find_and_tap_template
from src.core.network_sniffing import start_network_capture
from src.core.device import take_screenshot
from src.core.adb import get_screen_size, press_back
from src.game.controls import human_delay, humanized_tap, handle_swipes
from src.core.text_detection import (
    extract_text_from_region, 
    get_text_regions, 
    log_rejected_alliance,
    CONTROL_LIST,
)
from src.core.image_processing import (
    _save_debug_image
)
from src.core.audio import play_beep
import numpy as np

class SecretaryRemoveRoutine(TimeCheckRoutine):
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
        return self.remove_secretary()
    
    def remove_secretary(self) -> list[Tuple[int, int]]:
        try:
            firstlady = find_all_templates(
                self.device_id,
                "firstlady"
            )
            all_positions = {}
            for position_type in self.secretary_types:
                positions = find_all_templates(
                    self.device_id,
                    position_type
                )
                if positions:
                    all_positions[position_type] = positions[0]  # Take first match for each type
                    app_logger.debug(f"Found {position_type} position at ({positions[0][0]}, {positions[0][1]})")
            matches = find_all_templates(
                self.device_id,
                "clock"
            )
            sorted_matches = sorted(matches, key=lambda x: (x[1], x[0]))
            if sorted_matches:
                app_logger.debug(f"Found {len(sorted_matches)} accept buttons")
                app_logger.debug(f"Topmost button at coordinates: ({sorted_matches[0][0]}, {sorted_matches[0][1]})")
            screenshot = cv2.imread('tmp/screen.png')
            positions_to_process = {}
            gap_x = 50
            gap_y = 140
            for clock in sorted_matches:
                diff = (abs(clock[0] - firstlady[0][0]), abs(clock[1] - firstlady[0][1]))
                if diff[0] <= gap_x and diff[1] <= gap_y:
                    # 副大統領をスキップ
                    continue
                for position_type, pos_loc in all_positions.items():
                    pos_x, pos_y = pos_loc
                    x_diff = pos_x - clock[0]
                    y_diff = pos_y - clock[1]
                    if abs(x_diff) <= gap_x and abs(y_diff) <= gap_y:
                        positions_to_process[position_type] = {
                            "clock":(clock[0], clock[1]),
                            "position":(pos_loc)
                        }
            gap_to_time_x = 15
            gap_to_time_y = (-10)
            time_width = 78
            time_height = 20
            for name, entry in positions_to_process.items():
                text = ""
                x1 = entry['clock'][0] + gap_to_time_x
                y1 = entry['clock'][1] + gap_to_time_y
                width = time_width
                height = time_height
                x2 = x1 + width
                y2 = y1 + height
                cloppedImage = screenshot[y1 : y2, x1 : x2] # y, x

                cv2.rectangle(
                    screenshot, (x1, y1), (x2, y2), (0, 255, 0), 2
                )
                gray = cv2.cvtColor(np.array(cloppedImage), cv2.COLOR_BGR2GRAY)
                # Higher scale factor for better detail
                scale = 8
                enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                # Simple binary threshold
                # _, binary = cv2.threshold(enlarged, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                _save_debug_image(enlarged, f"{name}_clopped_time")
                config = (
                    '--psm 6 '
                    '--oem 3 '
                    f'-c tessedit_char_whitelist=:0123456789'
                )
                text = pytesseract.image_to_string(enlarged, lang='eng', config=config).strip()
                original_text = text
                text = text_sanitization(text)
                app_logger.info(f"### {name} original_text:{original_text} clean_up_text:{text}.")
                # Use regular expression to find time strings in HH:mm:ss format
                pattern = r'\b\d{2}:\d{2}:\d{2}\b'
                matches = re.findall(pattern, text)
                # Threshold in minutes
                threshold_minutes = CONFIG['thres_hold_minutes']
                if matches is None:
                    app_logger.info(f"{name} Screenshot returned NULL list.")
                elif not matches:
                    app_logger.info(f"{name} Screenshot returned no matches. Text: [{text.strip()}]")
                else:
                    # Threshold in minutes
                    total_minutes = time_to_minutes(matches[0])
                    app_logger.info(f"{name} [就任時間：{text.strip()}] 就任期間:{total_minutes}分.")
                    if total_minutes >= threshold_minutes:
                        app_logger.info(f"{name}を解任します [就任時間：{text.strip()}] 就任期間:{total_minutes}分.")
                        humanized_tap(self.device_id, entry['position'][0], entry['position'][1])
                        human_delay(CONFIG['timings']['menu_animation'])
                        width, height = get_screen_size(self.device_id)
                        dismiss = CONFIG['ui_elements']['secretary_dismiss_button']
                        dismiss_x = int(width * float(dismiss['x'].strip('%')) / 100)
                        dismiss_y = int(height * float(dismiss['y'].strip('%')) / 100)
                        app_logger.info(f"{name} [解任]ボタンをタップ.({dismiss_x},{dismiss_y})")
                        humanized_tap(
                            self.device_id,
                            dismiss_x,
                            dismiss_y
                        )
                        human_delay(CONFIG['timings']['menu_animation'])

                        confirm = CONFIG['ui_elements']['secretary_confirm_button']
                        confirm_x = int(width * float(confirm['x'].strip('%')) / 100)
                        confirm_y = int(height * float(confirm['y'].strip('%')) / 100)
                        app_logger.info(f"{name} [確認]ボタンをタップ.({confirm_x},{confirm_y})")
                        humanized_tap(
                            self.device_id,
                            confirm_x,
                            confirm_y
                        )
                        human_delay(CONFIG['timings']['menu_animation'])

                        press_back(self.device_id)
            _save_debug_image(screenshot, f"{name}_time")
            # cv2.imshow('temp', screenshot)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return True
        except Exception as e:
            app_logger.error(f"Error opening find_remove_secretary: {e}")
            return False
  
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
    
def time_to_minutes(time_str):
    hours, minutes, _ = map(int, time_str.split(':'))
    return hours * 60 + minutes

def text_sanitization(time_str):
    if not time_str:
        return ''
    if time_str[:3].isdigit():
        time_str = time_str[1:] # Remove extra leading digit
    parts = time_str.split(':')
    if len(parts) > 2 and len(parts[2]) > 2:
        parts[2] = parts[2][:2]  # Remove extra trailing digit
    return ':'.join(parts)

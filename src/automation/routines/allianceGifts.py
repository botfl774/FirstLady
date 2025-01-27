from src.automation.routines import TimeCheckRoutine
from src.core.adb import press_back
from src.core.config import CONFIG
from src.core.image_processing import find_and_tap_template
from src.game.controls import human_delay

class AllianceGiftsRoutine(TimeCheckRoutine):
    force_home: bool = True

    def _execute(self) -> bool:
        """Execute alliance donation sequence"""
        return self.execute_with_error_handling(self.collect_gifts)

    def collect_gifts(self) -> bool:
        self.automation.game_state["is_home"] = False

        """Navigate to the alliance donate menu and donate"""
        # Open alliance menu
        if not find_and_tap_template(
            self.device_id,
            "alliance",
            error_msg="Could not find alliance icon"
        ):
            return False
            
        # Click alliance tech icon
        if not find_and_tap_template(
            self.device_id,
            "alliance_gifts",
            error_msg="Could not find alliance_gifts icon"
        ):
            return False
            
        # Click collect all button
        if find_and_tap_template(
            self.device_id,
            "alliance_claim_all",
            error_msg="No claim all button found"
        ):
            # Clear claim message
            human_delay(CONFIG['timings']['menu_animation'])
            press_back(self.device_id)
            human_delay(CONFIG['timings']['menu_animation'])
        
        # Donate with long press
        if not find_and_tap_template(
            self.device_id,
            "alliance_gift_premium",
            error_msg="No premium tab found found"
        ):
            return True
        
        # Click collect all button
        find_and_tap_template(
            self.device_id,
            "alliance_claim_all",
            error_msg="No claim all button found"
        )
        
        return True
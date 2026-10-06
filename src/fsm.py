# fsm.py — Finite State Machine
# Manages robot task states: waiting, searching, gesture selection, navigation.
# Source: yolo_ros2/fsm_node.py (cleaned up, decoupled from ROS)
# Note: NAVIGATE_TO_BAG was left as `pass` in original — placeholder kept for now.

class FSM:
    # All possible states
    WAIT_START      = "WAIT_START"
    ANNOUNCE_READY  = "ANNOUNCE_READY"
    WAIT_GESTURE    = "WAIT_GESTURE"
    BAG_SELECTED    = "BAG_SELECTED"
    NAVIGATE_TO_BAG = "NAVIGATE_TO_BAG"
    DONE            = "DONE"

    def __init__(self):
        self.state = self.WAIT_START

        # Perception inputs (updated by controller)
        self.person_detected = False
        self.bag_detected = False

        # Gesture input
        self.gesture_result = None
        self.selected_bbox = None

        # Lock to prevent re-triggering
        self.bag_locked = False

        print(f"[FSM] Started in state: {self.state}")

    def update_vision(self, robot_state: dict):
        """Feed in latest YOLO state dict."""
        self.person_detected = robot_state.get("person_detected", False)
        self.bag_detected = robot_state.get("bag_detected", False)

    def update_gesture(self, gesture_result):
        """Feed in gesture result from GestureTracker (or None)."""
        if gesture_result and gesture_result.get("state") == "SELECT_BAG":
            self.gesture_result = gesture_result
            self.selected_bbox = gesture_result.get("bbox")

    def update_voice(self, action: str):
        """Feed in voice action parsed by Brain (e.g. START_SEARCH, STOP_MOTORS)."""
        print(f"[FSM] Voice action received: {action}")
        if action == "START_SEARCH" and self.state == self.WAIT_START:
            self.state = self.ANNOUNCE_READY
        elif action == "STOP_MOTORS":
            print("[FSM] Emergency stop!")
            self.state = self.WAIT_START

    def tick(self):
        """
        Advance the FSM by one step. Call this in your main loop.
        Returns an event string if something the controller needs to act on, else None.
        Events: 'ANNOUNCE_READY', 'BAG_LOCKED', 'NAVIGATE', None
        """
        event = None

        if self.state == self.WAIT_START:
            # WAIT_START no longer auto-transitions on vision alone.
            # Transition to ANNOUNCE_READY must be triggered by voice (START_SEARCH)
            pass

        elif self.state == self.ANNOUNCE_READY:
            print("[FSM] System ready — announcing.")
            event = 'ANNOUNCE_READY'
            self.state = self.WAIT_GESTURE

        elif self.state == self.WAIT_GESTURE:
            if self.gesture_result is not None:
                self.state = self.BAG_SELECTED

        elif self.state == self.BAG_SELECTED:
            if not self.bag_locked and self.selected_bbox is not None:
                self.bag_locked = True
                print(f"[FSM] Bag locked at bbox={self.selected_bbox}")
                event = 'BAG_LOCKED'
                self.state = self.NAVIGATE_TO_BAG

        elif self.state == self.NAVIGATE_TO_BAG:
            # TODO: implement navigation logic
            # For now, signal controller to handle it
            event = 'NAVIGATE'

        return event

    def get_state(self):
        return self.state
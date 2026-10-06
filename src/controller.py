# controller.py — Main Controller
# Orchestrates all modules: speech pipeline, YOLO, gesture, FSM, brain.
# This is the "brain stem" that decides what runs and when.
#
# Run order:
#   1. YOLO + Gesture run continuously in background (camera loop)
#   2. FSM ticks every loop to track task state
#   3. Speech pipeline (ear → recorder → scribe → brain → mouth) triggers on wake word
#   4. Brain is kept aware of robot vision state at all times

import os
import sys
import json
import threading
import time
import cv2

# Add src/ to path so imports work
sys.path.insert(0, os.path.dirname(__file__))

from ear import Ear
from recorder import Recorder
from scribe import Scribe
from brain import Brain
from mouth import Mouth
from fsm import FSM

# Vision modules — comment these out if running without a camera/YOLO model
try:
    from yolo_tracker import YoloTracker
    from gesture_tracker import GestureTracker
    VISION_ENABLED = True
except Exception as e:
    print(f"[CONTROLLER] Vision modules not loaded: {e}")
    VISION_ENABLED = False


class Controller:
    def __init__(self):
        print("--- JARVIS ROBOT STARTUP ---")

        # Speech pipeline
        self.ear      = Ear()
        self.recorder = Recorder()
        self.scribe   = Scribe()
        self.brain    = Brain()
        self.mouth    = Mouth()

        # Task FSM
        self.fsm = FSM()

        # Vision (optional)
        if VISION_ENABLED:
            self.yolo    = YoloTracker()
            self.gesture = GestureTracker()

        self.latest_robot_state = {}
        self.latest_cmd = {'linear_x': 0.0, 'angular_z': 0.0}
        self.running = True

        print("--- READY ---")

    # --------------------------------------------------
    # SPEECH LOOP (runs in its own thread)
    # Waits for wake word → records → transcribes → thinks → speaks
    # --------------------------------------------------
    def speech_loop(self):
        while self.running:
            if self.ear.wait_for_wake_word():
                # If a person and a bag are currently detected, use the wake word
                # to start the selection flow instead of the normal voice command flow.
                vision = self.latest_robot_state
                person = vision.get('person_detected', False)
                bag = vision.get('bag_detected', False)

                if person and bag:
                    # Trigger FSM to announce readiness for gesture selection
                    # and greet the user.
                    self.fsm.update_voice("START_SEARCH")
                    self.mouth.speak("hi what can i help u")
                    # Do not proceed to record/transcribe — waiting for gesture.
                    continue

                # Default: normal voice command flow
                self.mouth.speak("Yes?")

                filename = self.recorder.record("command.wav")
                if not filename:
                    continue

                user_text = self.scribe.listen(filename)
                print(f"[USER]: {user_text}")

                # Give brain latest robot vision context before thinking
                self.brain.update_vision(self.latest_robot_state)
                ai_reply = self.brain.think(user_text)

                if ai_reply:
                    print(f"[JARVIS]: {ai_reply}")

                    # Try to parse JSON first — brain may return structured response
                    # If JSON: speak only the "response" field, pass "action" to FSM
                    # If plain text: speak it directly
                    try:
                        parsed = json.loads(ai_reply)
                        spoken = parsed.get("response")
                        action = parsed.get("action")
                        if spoken:
                            self.mouth.speak(spoken)
                        if action:
                            self.fsm.update_voice(action)
                    except (json.JSONDecodeError, TypeError):
                        # Plain text reply — speak as-is
                        self.mouth.speak(ai_reply)

                # Cleanup audio file
                if os.path.exists("command.wav"):
                    os.remove("command.wav")

    # --------------------------------------------------
    # VISION LOOP (runs in its own thread)
    # Continuously processes camera frames through YOLO + gesture
    # --------------------------------------------------
    def vision_loop(self, get_frames_fn):
        """
        get_frames_fn: a callable that returns (rgb_frame, depth_frame).
        Hook this up to your RealSense / ROS camera feed.
        """
        while self.running:
            rgb, depth = get_frames_fn()
            if rgb is None:
                time.sleep(0.01)
                continue

            # YOLO person tracking
            cmd, state = self.yolo.process(rgb, depth)
            self.latest_robot_state = state
            self.latest_cmd = cmd

            # Gesture bag selection
            gesture_result = self.gesture.process(rgb)
            if gesture_result:
                self.fsm.update_gesture(gesture_result)
                state['bag_detected'] = True

            # FSM vision update
            self.fsm.update_vision(state)

            # --- Preview / Visualization ---
            try:
                disp = rgb.copy()
                # Draw YOLO boxes if available
                if hasattr(self, 'yolo') and getattr(self.yolo, 'latest_boxes', None) is not None:
                    for box in self.yolo.latest_boxes:
                        try:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                        except Exception:
                            continue
                        cls = int(box.cls[0]) if hasattr(box, 'cls') else -1
                        tid = int(box.id[0]) if getattr(box, 'id', None) is not None else -1
                        if cls == config.YOLO_PERSON_CLASS:
                            color = (255, 100, 0)
                            label = 'person'
                        elif cls == config.YOLO_BAG_CLASS:
                            color = (0, 0, 255)
                            label = 'bag'
                        else:
                            color = (200, 200, 200)
                            label = str(cls)
                        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
                        text = f"{label}"
                        if tid >= 0:
                            text += f" id={tid}"
                        cv2.putText(disp, text, (max(0, x1), max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Status text
                status = state.get('state', '')
                info = f"{status} | person={state.get('person_detected')} bag={state.get('bag_detected')}"
                cv2.putText(disp, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                cv2.imshow('Jarvis Preview', disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
            except Exception:
                pass

    # --------------------------------------------------
    # FSM LOOP (runs in its own thread)
    # Ticks the state machine and reacts to events
    # --------------------------------------------------
    def fsm_loop(self):
        while self.running:
            event = self.fsm.tick()

            if event == 'ANNOUNCE_READY':
                # ANNOUNCE_READY handled via speech_loop greeting; keep for compatibility
                self.mouth.speak("I can see a person and a bag. Please point at the bag you want me to carry.")

            elif event == 'BAG_LOCKED':
                # New message per user request
                self.mouth.speak("I see you selected the bag")
                # TODO: trigger your navigation system here

            elif event == 'NAVIGATE':
                # TODO: publish cmd_vel or trigger nav stack
                pass

            time.sleep(0.2)

    # --------------------------------------------------
    # START
    # --------------------------------------------------
    def start(self, get_frames_fn=None):
        # Speech runs in background thread
        speech_thread = threading.Thread(target=self.speech_loop, daemon=True)
        speech_thread.start()

        # FSM runs in background thread
        fsm_thread = threading.Thread(target=self.fsm_loop, daemon=True)
        fsm_thread.start()

        # Vision loop — only start if VISION_ENABLED and camera feed provided
        if VISION_ENABLED and get_frames_fn is not None:
            vision_thread = threading.Thread(
                target=self.vision_loop,
                args=(get_frames_fn,),
                daemon=True
            )
            vision_thread.start()
        elif not VISION_ENABLED:
            print("[CONTROLLER] Running without vision (speech only mode).")

        print("[CONTROLLER] All systems running. Press Ctrl+C to stop.")
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[CONTROLLER] Shutting down.")
            self.running = False


if __name__ == "__main__":
    bot = Controller()
    bot.start()  # starts in speech-only mode (no camera feed passed in)
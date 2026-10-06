# brain.py — LLM Brain
# Thinks and generates a response using Ollama.
# Combines SERVICE-BOT-AI memory system + yolo_ros2's robot vision context awareness.
# Fixes: missing os import bug from yolo_ros2's brain_node.py

import os
import json
import ollama
import config


class Brain:
    def __init__(self):
        self.model = config.OLLAMA_MODEL
        self.memory_file = config.MEMORY_FILE
        self.latest_robot_vision = {}   # updated by yolo/fsm state

        # Load saved memory if it exists, otherwise start fresh
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    self.messages = json.load(f)
                print(f"[BRAIN] Loaded memory from {self.memory_file}")
            except Exception:
                self.messages = self._default_memory()
        else:
            self.messages = self._default_memory()

        print(f"[BRAIN] Online: {self.model}")

    def _default_memory(self):
        return [{'role': 'system', 'content': config.SYSTEM_PROMPT}]

    def update_vision(self, robot_state: dict):
        """Called by the controller with latest YOLO/FSM state."""
        self.latest_robot_vision = robot_state

    def save_memory(self):
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.messages, f, indent=4)
        except Exception as e:
            print(f"[BRAIN ERROR] Memory save failed: {e}")

    def think(self, user_text):
        if not user_text:
            return None

        # Inject current robot vision state into user message
        user_content = user_text
        if self.latest_robot_vision:
            user_content = json.dumps({
                "user_instruction": user_text,
                "current_robot_state": self.latest_robot_vision
            })

        self.messages.append({'role': 'user', 'content': user_content})
        print(f"[BRAIN] Thinking...")

        try:
            response = ollama.chat( model=self.model, messages=self.messages )
            reply = response['message']['content']

            self.messages.append({'role': 'assistant', 'content': reply})

            # Trim memory — keep system prompt + last N turns
            if len(self.messages) > (config.MEMORY_LIMIT + 1):
                self.messages = [self.messages[0]] + self.messages[-config.MEMORY_LIMIT:]

            self.save_memory()
            return reply

        except Exception as e:
            print(f"[BRAIN ERROR] {e}")
            return "Sorry, I couldn't process that."
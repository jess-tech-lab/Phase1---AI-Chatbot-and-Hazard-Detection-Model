import subprocess
import re

class Processor:
    """
    Simple command processor for handling recognized voice text.
    We can expand this to include NLP intent matching, regex parsing,
    or AI model calls (like OpenAI or local LLM).
    """

    def __init__(self):
        # Initialize configurations or model references here
        self.commands = {
            "hazard": self.run_hazard_detection,
            "detect": self.run_hazard_detection,
            "stop": self.stop_hazard_detection,
            "hello": self.greet_user
        }

    async def process_command(self, text: str) -> str:
        """
        Process the transcribed voice command.
        Args:
            text (str): the transcribed speech text
        Returns:
            str: the response text
        """

        # Normalize text (basic cleanup)
        cleaned = text.lower().strip()
        print(f"[Processor] Received command: {cleaned}")

        # Match known commands
        for keyword, action in self.commands.items():
            if re.search(rf"\b{keyword}\b", cleaned):
                print(f"[Processor] Matched command: {keyword}")
                return await action(cleaned)

        # Default fallback if no command recognized
        print("[Processor] Unknown command. No action taken.")
        return "Sorry, I didn’t recognize that command."

    # ---------------------------------------------------------
    # Actions below (can customize later)
    # ---------------------------------------------------------

    async def run_hazard_detection(self, text: str) -> str:
        """Run YOLO hazard detection as a subprocess."""
        print("[Processor] Running hazard detection…")
        subprocess.Popen(
            ["python", "hazard_detection.py", "--model", "yolov8n.pt"]
        )
        return "Hazard detection started."

    async def stop_hazard_detection(self, text: str) -> str:
        """Example stop function — you can later add process tracking."""
        print("[Processor] Stopping hazard detection (simulated).")
        # Add real logic for terminating a subprocess if needed
        return "Hazard detection stopped."

    async def greet_user(self, text: str) -> str:
        """Simple greeting response."""
        print("[Processor] Greeting user.")
        return "Hello there! How can I assist you today?"
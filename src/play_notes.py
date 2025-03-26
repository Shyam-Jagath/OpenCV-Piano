import pygame
import os

class SoundPlayer:
    def __init__(self, notes):
        self.notes = notes
        self.sounds = {}
        self.base_path = os.path.join(os.path.dirname(__file__), "..", "resources", "sounds")
        self.load_sounds()

    def load_sounds(self):
        # Force SDL to use PulseAudio (Change to 'alsa' if needed)
        os.environ['SDL_AUDIODRIVER'] = 'pulseaudio'
        
        # Initialize the pygame mixer
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            print("Pygame mixer initialized successfully!")
        except pygame.error as e:
            print(f"Pygame mixer failed to initialize: {e}")
            return

        for note in self.notes:
            sound_path = os.path.join(self.base_path, f"{note}.wav")  # Using .wav format
            if os.path.exists(sound_path):
                try:
                    self.sounds[note] = pygame.mixer.Sound(sound_path)
                    print(f"Loaded: {sound_path}")
                except pygame.error as e:
                    print(f"Failed to load {sound_path}: {e}")
            else:
                print(f"Warning: {sound_path} not found.")

    def change_sounds_folder(self, new_folder):
        self.base_path = new_folder
        self.sounds.clear()
        self.load_sounds()
        print(f"Sounds folder changed to: {new_folder}")

    def play_note_by_index(self, i):
        if 0 <= i < len(self.notes):
            note = self.notes[i]
            if note in self.sounds:
                print(f"🎶 Playing: {note}")
                self.sounds[note].play()
            else:
                print(f"Sound not loaded for {note}.")
        else:
            print(f"Invalid index: {i}")

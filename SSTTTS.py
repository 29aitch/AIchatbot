import speech_recognition as sr
import pyttsx3
import json
import os


class MultilingualSpeechTest:
    def __init__(self):
        # Speech Recognition Setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Text-to-Speech Setup
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Adjust speech rate

        # Supported languages dictionary
        self.supported_languages = {
            'en': 'English',
            'zh-CN': 'Simplified Chinese',
            'zh-TW': 'Traditional Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'ru': 'Russian',
            'ar': 'Arabic',
            'hi': 'Hindi'
        }

    def list_languages(self):
        """List available languages for speech recognition and TTS"""
        print("\nSupported Languages:")
        for code, name in self.supported_languages.items():
            print(f"{code}: {name}")

    def recognize_speech(self, language='en'):
        """
        Recognize speech from microphone input in specified language.

        Args:
            language (str): Language code for speech recognition

        Returns:
            tuple: (recognized text, language code)
        """
        with self.microphone as source:
            print(f"Adjusting for ambient noise in {self.supported_languages.get(language, language)}. Please wait...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

            print(f"Listening in {self.supported_languages.get(language, language)}... Speak now.")
            try:
                audio = self.recognizer.listen(source, timeout=5)

                try:
                    text = self.recognizer.recognize_google(audio, language=language)
                    print(f"Recognized [{language}]: {text}")
                    return text, language
                except sr.UnknownValueError:
                    print(f"Could not understand audio in {language}")
                except sr.RequestError as e:
                    print(f"Could not request results in {language}; {e}")

            except sr.WaitTimeoutError:
                print("Listening timed out while waiting for speech.")

        return None, language

    def text_to_speech(self, text, language='en'):
        """
        Convert text to speech in specified language.

        Args:
            text (str): Text to convert to speech
            language (str): Language code for speech synthesis
        """
        print(f"\nConverting text to speech in {self.supported_languages.get(language, language)}")
        print(f"Text: {text}")

        # Check for language-specific voices
        voices = self.tts_engine.getProperty('voices')
        target_voice = None

        # Attempt to find a voice that matches the language
        for voice in voices:
            # Different TTS engines have different voice language detection
            # This is a simplistic approach and may need adjustments
            if language.lower() in voice.languages or language.lower() in voice.name.lower():
                target_voice = voice
                break

        if target_voice:
            self.tts_engine.setProperty('voice', target_voice.id)

        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Error in text-to-speech: {e}")

    def interactive_test(self):
        """Interactive multilingual speech test"""
        while True:
            self.list_languages()

            # Language selection
            language = input("\nEnter language code (or 'exit' to quit): ").strip()

            if language.lower() in ['exit', 'quit', 'bye']:
                break

            if language not in self.supported_languages:
                print("Invalid language code. Please try again.")
                continue

            # Speech Recognition
            text, lang = self.recognize_speech(language)

            if text:
                # Text-to-Speech
                self.text_to_speech(text, lang)


def main():
    speech_test = MultilingualSpeechTest()
    speech_test.interactive_test()


if __name__ == "__main__":
    main()
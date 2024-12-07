import json

import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import os
import tempfile
import pygame
from pathlib import Path
from typing import Annotated, Union, Optional
import typer
from peft import PeftModelForCausalLM
from transformers import AutoModel, AutoTokenizer
import torch

# Import the store items from the previous artifact
from store_inventory import store_items

app = typer.Typer(pretty_exceptions_show_locals=False)


class RetailStoreAssistant:
    supported_languages = {
        'en-US': {
            'name': 'English (US)',
            'welcome': 'Welcome to our store! How can I help you today?',
            'not_found': 'I apologize, I could not find information about that item.',
            'unrelated_topic': 'I am designed to help you with store-related queries. How can I assist you in our store today?',
            'noise_prompt': 'Adjusting for ambient noise. Please wait...',
            'listen_prompt': 'Listening... Please speak clearly.',
            'exit_phrases': ['goodbye', 'bye', 'exit', 'quit', 'see you later'],
            'exit_response': 'Goodbye! Returning to language selection.',
            'ready_prompt': 'Ready for your next question. How can I help you?',
            'tts_voice': 'english',
            'inventory_menu': '9. Manage Store Inventory',
            'enter_password': 'Please enter the password.',
            'incorrect_password': 'Incorrect password. Access denied.',
            'inventory_options': 'Choose an option:\n1. Create Item\n2. Delete Item',
            'enter_item_name': 'Please tell me the name of the item.',
            'enter_floor': 'What floor is the item located on?',
            'enter_aisle': 'In which aisle is the item placed?',
            'confirm_delete': 'Are you sure you want to delete this item? Say yes or no.',
            'item_created': 'Item successfully created.',
            'item_deleted': 'Item successfully deleted.',
            'item_not_found': 'Item not found in the inventory.'
        },
        'zh-CN': {
            'name': 'Chinese (Simplified)',
            'welcome': '欢迎来到我们的商店！今天我可以为您做些什么？',
            'not_found': '抱歉，我找不到关于该物品的信息。',
            'unrelated_topic': '我专门设计用来帮助与商店相关的查询。今天我能为您做些什么？',
            'noise_prompt': '正在调整环境噪音。请稍等...',
            'listen_prompt': '正在聆听...请清晰地说话。',
            'ready_prompt': '准备好听取您的下一个问题。我能帮您什么？',
            'exit_phrases': ['再见', '拜拜', '退出', '结束', '离开'],
            'exit_response': '再见！正在返回语言选择页面。',
            'tts_voice': 'chinese'
        },
        'ms-MY': {
            'name': 'Malay',
            'welcome': 'Selamat datang ke kedai kami! Bagaimana saya boleh membantu anda hari ini?',
            'not_found': 'Maaf, saya tidak dapat menemui maklumat tentang item tersebut.',
            'unrelated_topic': 'Saya direka untuk membantu anda dengan pertanyaan berkaitan kedai. Bagaimana saya boleh membantu anda hari ini?',
            'noise_prompt': 'Menyesuaikan untuk bunyi persekitaran. Sila tunggu...',
            'listen_prompt': 'Mendengar... Sila bercakap dengan jelas.',
            'exit_phrases': ['selamat tinggal', 'jumpa lagi', 'keluar', 'berhenti', 'sampai jumpa'],
            'exit_response': 'Selamat tinggal! Kembali ke pilihan bahasa.',
            'ready_prompt': 'Sedia untuk soalan anda yang seterusnya. Bagaimana saya boleh membantu anda?',
            'tts_voice': 'malay'
        },
        'fr-FR': {
            'name': 'French',
            'welcome': 'Bienvenue dans notre magasin! Comment puis-je vous aider aujourd\'hui?',
            'not_found': 'Je suis désolé, je n\'ai pas pu trouver d\'informations sur cet article.',
            'unrelated_topic': 'Je suis conçu pour vous aider avec des questions liées au magasin. Comment puis-je vous aider aujourd\'hui?',
            'noise_prompt': 'Ajustement au bruit ambiant. Veuillez patienter...',
            'listen_prompt': 'À l\'écoute... Veuillez parler clairement.',
            'ready_prompt': 'Prêt pour votre prochaine question. Comment puis-je vous aider?',
            'exit_phrases': ['au revoir', 'salut', 'bye', 'quitter', 'partir'],
            'exit_response': 'Au revoir ! Retour à la sélection de langue.',
            'tts_voice': 'french'
        },
        'es-ES': {
            'name': 'Spanish',
            'welcome': '¡Bienvenido a nuestra tienda! ¿Cómo puedo ayudarle hoy?',
            'not_found': 'Lo siento, no pude encontrar información sobre ese artículo.',
            'unrelated_topic': 'Estoy diseñado para ayudarle con consultas relacionadas con la tienda. ¿Cómo puedo ayudarle hoy?',
            'noise_prompt': 'Ajustando el ruido ambiental. Por favor, espere...',
            'listen_prompt': 'Escuchando... Por favor, hable claramente.',
            'exit_phrases': ['adiós', 'hasta luego', 'chao', 'salir', 'terminar'],
            'exit_response': '¡Adiós! Volviendo a la selección de idioma.',
            'ready_prompt': 'Listo para su próxima pregunta. ¿Cómo puedo ayudarle?',
            'tts_voice': 'spanish'
        }
    }

    def __init__(self, model_dir: Union[str, Path], language: str = 'en-US'):
        # Speech Recognition Setup with longer pause threshold
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.5  # Increased pause threshold to prevent cutting off
        self.microphone = sr.Microphone()

        # Language Configuration
        self.language = language

        # Model and Tokenizer Setup
        self.model, self.tokenizer = self._load_model_and_tokenizer(model_dir)

        # Text-to-Speech Setup
        pygame.mixer.init()  # Initialize pygame mixer for better audio playback
        self.store_items = store_items
        self.inventory_path = Path(__file__).parent / 'store_inventory.py'

    def _save_inventory(self):
        """
        Save the updated inventory back to the store_inventory.py file
        """
        try:
            with open(self.inventory_path, 'w', encoding='utf-8') as f:
                f.write("store_items = " + json.dumps(self.store_items, indent=4))

            # Also update the imported store_items
            global store_items
            store_items = self.store_items
        except Exception as e:
            print(f"Error saving inventory: {e}")

    def _generate_directions(self, destination):
        """
        Generate fictional directions and distance for various destinations
        """
        # Predefined fictional locations with random distances
        location_details = {
            # Shopping Malls
            "westfield mall": {
                "directions": "Head east on Main Street, turn right at Oak Avenue. The mall will be on your left.",
                "distance": "2.3 miles"
            },
            "city center mall": {
                "directions": "Take Highway 101 south, exit at Maple Street. Follow signs to City Center Mall.",
                "distance": "5.7 miles"
            },
            # Shops
            "downtown bookstore": {
                "directions": "Go north on River Road, turn left at Elm Street. The bookstore is in the second block.",
                "distance": "1.5 miles"
            },
            "corner bakery": {
                "directions": "Head west on Central Avenue, turn right at Pine Street. You'll see the bakery on the corner.",
                "distance": "0.8 miles"
            },
            # Gas Stations
            "quick fill gas station": {
                "directions": "Take the interstate exit 45, the gas station is right off the highway on your right.",
                "distance": "3.2 miles"
            },
            "city gas and go": {
                "directions": "Follow Market Street towards downtown, the gas station is near the intersection with Broadway.",
                "distance": "4.1 miles"
            }
        }

        # Normalize destination for matching
        dest_lower = destination.lower()

        # Check for partial matches
        for key, details in location_details.items():
            if key in dest_lower:
                return (f"To reach the {key.title()}: {details['directions']} "
                        f"The distance is approximately {details['distance']}.")

        # Fallback response for unknown destinations
        return ("I apologize, but I don't have specific directions for that location. "
                "GPS integration will be coming soon.")
    def _manage_inventory(self):
        """
        Manage store inventory with password protection
        """
        language_config = self.supported_languages.get(self.language, self.supported_languages['en-US'])

        # Password check (text-based)
        print(language_config['enter_password'])
        password_attempt = input("Enter password: ")

        if password_attempt.strip() != '1234':
            print(language_config['incorrect_password'])
            return  # Return to main menu instead of breaking the entire flow

        # Inventory management options (text-based)
        print(language_config['inventory_options'])

        while True:
            choice = input("Enter your choice (1/2): ").strip()

            if choice == '1':
                self._create_item()
                break
            elif choice == '2':
                self._delete_item()
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")

    def _create_item(self):
        """
        Create a new item in the inventory
        """
        language_config = self.supported_languages.get(self.language, self.supported_languages['en-US'])

        # Get item name (text-based)
        print(language_config['enter_item_name'])
        item_name = input("Item Name: ")

        # Get floor
        print(language_config['enter_floor'])
        floor = input("Floor: ")

        # Get aisle
        print(language_config['enter_aisle'])
        aisle = input("Aisle: ")

        # Generate description using the model
        description_prompt = f"Generate a concise description for a store item called {item_name} located in the {aisle} on the {floor}. Highlight its key features and potential uses."
        description = self.generate_response(description_prompt)

        # Create the new item
        new_item = [item_name, floor, aisle, description]
        self.store_items.append(new_item)

        # Save the updated inventory
        self._save_inventory()

        # Confirm creation
        print(language_config['item_created'])

    def _delete_item(self):
        """
        Delete an item from the inventory
        """
        language_config = self.supported_languages.get(self.language, self.supported_languages['en-US'])

        # Get item name to delete
        print(language_config['enter_item_name'])
        item_name = input("Item Name: ")

        # Find the item
        item_to_delete = None
        for item in self.store_items:
            if item_name.lower() == item[0].lower():
                item_to_delete = item
                break

        if not item_to_delete:
            print(language_config['item_not_found'])
            return

        # Confirm deletion
        print(language_config['confirm_delete'])
        confirmation = input("Confirm deletion (yes/no): ").lower()

        if confirmation == 'yes':
            self.store_items.remove(item_to_delete)

            # Save the updated inventory
            self._save_inventory()

            print(language_config['item_deleted'])

    def _check_exit_command(self, text):
        """
        Check if the input text matches any exit phrases for the current language
        """
        language_config = self.supported_languages.get(self.language, self.supported_languages['en-US'])
        exit_phrases = language_config['exit_phrases']

        # Convert input to lowercase for case-insensitive matching
        text_lower = text.lower().strip()

        # Check if text matches any exit phrase
        return any(exit_phrase in text_lower for exit_phrase in exit_phrases)

    def text_to_speech(self, text):
        """
        Convert text to speech using multilingual support with gTTS
        """
        language_config = self.supported_languages.get(self.language, self.supported_languages['en-US'])
        language_code = self.language.split('-')[0]  # Extract language code

        try:
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
                # Use gTTS for multilingual text-to-speech
                tts = gTTS(text=text, lang=language_code, slow=False)
                tts.save(temp_audio.name)

                # Play the audio using pygame
                pygame.mixer.music.load(temp_audio.name)
                pygame.mixer.music.play()

                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

                # Clean up the temporary file
                pygame.mixer.music.unload()
                os.unlink(temp_audio.name)

        except Exception as e:
            print(f"Text-to-Speech Error: {e}")
            print(f"Fallback: Printing text: {text}")

    def _find_item_details(self, item_name):
        """Enhanced item search with multiple matching strategies"""
        item_name = item_name.lower()

        # Exact match first
        for item in self.store_items:
            if item_name == item[0].lower():
                return item

        # Partial match
        for item in self.store_items:
            if item_name in item[0].lower():
                return item

        return None

    def generate_response(self, text):
        """
        Enhanced response generation to include direction queries
        """
        # Check for direction-related queries first
        direction_keywords = ['directions', 'how to get to', 'way to', 'route to']
        if any(keyword in text.lower() for keyword in direction_keywords):
            # Extract destination by removing direction keywords
            destination = text.lower()
            for keyword in direction_keywords:
                destination = destination.replace(keyword, '').strip()

            return self._generate_directions(destination)

        # Existing item location detection
        item_details = self._find_item_details(text)
        if item_details:
            response = (f"{item_details[0]} is located on the {item_details[1]} in {item_details[2]}. "
                        f"Description: {item_details[3]}")
            return response

        # Prepare messages for model with store context
        context = "Store Inventory: " + " | ".join([
            f"{item[0]} in {item[1]} at {item[2]}: {item[3]}"
            for item in self.store_items
        ])

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful Retail Store Assistant. Your primary goal is to "
                    "assist customers with store-related queries. Be friendly, polite, "
                    "and provide helpful information about products, locations, and services. "
                    f"Current Store Context: {context}"
                ),
            },
            {"role": "user", "content": text}
        ]

        # Rest of the model response generation remains the same as in the original implementation
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.model.device)

        generate_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "top_p": 0.8,
            "temperature": 0.8,
            "repetition_penalty": 1.2,
            "eos_token_id": self.model.config.eos_token_id,
        }

        outputs = self.model.generate(**inputs, **generate_kwargs)
        response = self.tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        ).strip()

        return response

    def _load_model_and_tokenizer(self, model_dir: Union[str, Path]):
        # Previous implementation remains the same as the original code
        model_dir = Path(model_dir).expanduser().resolve()
        if (model_dir / 'adapter_config.json').exists():
            import json
            with open(model_dir / 'adapter_config.json', 'r', encoding='utf-8') as file:
                config = json.load(file)
            model = AutoModel.from_pretrained(
                config.get('base_model_name_or_path'),
                trust_remote_code=True,
                device_map='auto',
                torch_dtype=torch.float16
            )
            model = PeftModelForCausalLM.from_pretrained(
                model=model,
                model_id=model_dir,
                trust_remote_code=True,
            )
            tokenizer_dir = model.peft_config['default'].base_model_name_or_path
        else:
            model = AutoModel.from_pretrained(
                model_dir,
                trust_remote_code=True,
                device_map='auto',
                torch_dtype=torch.float16
            )
            tokenizer_dir = model_dir

        model = model.to("mps" if torch.backends.mps.is_available() else "auto")

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            trust_remote_code=True,
            encode_special_tokens=True,
            use_fast=False
        )
        return model, tokenizer

    # Rest of the code remains the same as the original implementation
    def recognize_speech(self):
        """
        Recognize speech from microphone input in specified language.
        Uses a longer dynamic energy threshold to prevent cutting off.
        """
        with self.microphone as source:
            language_config = self.supported_languages.get(self.language, self.supported_languages['en-US'])

            print(language_config['noise_prompt'])
            # Dynamic energy threshold helps prevent cutting off
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

            print(language_config['listen_prompt'])
            # Longer timeout to allow for longer phrases
            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)

        try:
            text = self.recognizer.recognize_google(audio, language=self.language)
            print(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")

        return ""

    def run_continuous_pipeline(self):
        """
        Continuous speech recognition with language selection and exit command
        """
        while True:
            try:
                # Display language options
                print("Please select a language:")
                languages = list(self.supported_languages.keys())
                for i, lang in enumerate(languages, 1):
                    print(f"{i}. {self.supported_languages[lang]['name']}")

                # Add inventory management option
                print(self.supported_languages['en-US']['inventory_menu'])

                # Get language selection
                while True:
                    try:
                        choice = input("Enter the number of your preferred language or option: ")
                        choice = int(choice)

                        if 1 <= choice <= len(languages):
                            self.language = languages[choice - 1]
                            break
                        elif choice == len(languages) + 1:
                            self._manage_inventory()
                            continue
                        else:
                            print("Invalid choice. Please try again.")
                    except ValueError:
                        print("Please enter a valid number.")

                # Welcome message
                language_config = self.supported_languages.get(self.language, self.supported_languages['en-US'])
                welcome_msg = language_config['welcome']
                print(welcome_msg)
                self.text_to_speech(welcome_msg)

                # Inner conversation loop
                while True:
                    # Speech Recognition
                    print(language_config['ready_prompt'])
                    input_text = self.recognize_speech()

                    if not input_text:
                        continue

                    # Check for exit command
                    if self._check_exit_command(input_text):
                        # Exit response
                        exit_msg = language_config['exit_response']
                        print(exit_msg)
                        self.text_to_speech(exit_msg)
                        break  # Break inner loop to return to language selection

                    # Generate and output response
                    response = self.generate_response(input_text)
                    print(f"Assistant: {response}")
                    self.text_to_speech(response)

            except KeyboardInterrupt:
                print("\nAssistant stopped by user.")
                break


@app.command()
def main(
        model_dir: Annotated[str, typer.Argument(help='Path to the model directory')],
        language: Annotated[Optional[str], typer.Option(help='Initial language code')] = 'en-US'
):
    pipeline = RetailStoreAssistant(model_dir, language)
    pipeline.run_continuous_pipeline()

if __name__ == '__main__':
    app()
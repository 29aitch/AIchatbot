import speech_recognition as sr
import pyttsx3
from pathlib import Path
from typing import Annotated, Union, Optional
import typer
from peft import PeftModelForCausalLM
from transformers import AutoModel, AutoTokenizer
import torch

app = typer.Typer(pretty_exceptions_show_locals=False)


class SpeechInferencePipeline:
    # Move supported_languages to be a class attribute
    supported_languages = {
        'en-US': {
            'name': 'English (US)',
            'noise_prompt': 'Adjusting for ambient noise. Please wait...',
            'listen_prompt': 'Listening... Speak now.',
            'ready_prompt': 'Ready for next input. Speak now or say "exit" to quit.'
        },
        'zh-CN': {
            'name': 'Chinese (Simplified)',
            'noise_prompt': '正在调整环境噪音。请稍等...',
            'listen_prompt': '正在聆听...请说话。',
            'ready_prompt': '准备接收下一个输入。请说话，或说"退出"以结束。'
        },
        'ms-MY': {
            'name': 'Malay (Malaysia)',
            'noise_prompt': 'Menyesuaikan bunyi sekitar. Sila tunggu...',
            'listen_prompt': 'Sedang mendengar... Sila bercakap.',
            'ready_prompt': 'Sedia untuk input seterusnya. Sila bercakap atau sebut "keluar" untuk berhenti.'
        },
        'es-ES': {
            'name': 'Spanish (Spain)',
            'noise_prompt': 'Ajustando el ruido ambiente. Por favor, espere...',
            'listen_prompt': 'Escuchando... Hable ahora.',
            'ready_prompt': 'Listo para el siguiente input. Hable ahora o diga "salir" para terminar.'
        },
        'fr-FR': {
            'name': 'French (France)',
            'noise_prompt': 'Ajustement du bruit ambiant. Veuillez patienter...',
            'listen_prompt': 'À l\'écoute... Parlez maintenant.',
            'ready_prompt': 'Prêt pour la prochaine entrée. Parlez ou dites "sortir" pour quitter.'
        },
        'de-DE': {
            'name': 'German (Germany)',
            'noise_prompt': 'Umgebungsgeräusche werden angepasst. Bitte warten...',
            'listen_prompt': 'Höre zu... Sprechen Sie jetzt.',
            'ready_prompt': 'Bereit für die nächste Eingabe. Sprechen Sie oder sagen Sie "beenden" zum Aufhören.'
        },
        'ja-JP': {
            'name': 'Japanese',
            'noise_prompt': '周囲の雑音を調整しています。お待ちください...',
            'listen_prompt': '聞いています...今話してください。',
            'ready_prompt': '次の入力の準備ができました。話すか、「終了」と言って終了してください。'
        },
        'ko-KR': {
            'name': 'Korean',
            'noise_prompt': '주변 소음을 조정하고 있습니다. 잠시만 기다려 주세요...',
            'listen_prompt': '듣고 있습니다... 이제 말씀해 주세요.',
            'ready_prompt': '다음 입력 준비가 되었습니다. 말씀하거나 "종료"라고 말씀해 주세요.'
        },
        'ru-RU': {
            'name': 'Russian',
            'noise_prompt': 'Настройка окружающего шума. Пожалуйста, подождите...',
            'listen_prompt': 'Слушаю... Говорите сейчас.',
            'ready_prompt': 'Готов к следующему вводу. Говорите или скажите "выход", чтобы закончить.'
        },
        'ar-SA': {
            'name': 'Arabic (Saudi Arabia)',
            'noise_prompt': 'جارٍ ضبط الضوضاء المحيطة. الرجاء الانتظار...',
            'listen_prompt': 'أستمع... تحدث الآن.',
            'ready_prompt': 'جاهز للإدخال التالي. تحدث الآن أو قل "خروج" للإنهاء.'
        },
        'hi-IN': {
            'name': 'Hindi (India)',
            'noise_prompt': 'पर्यावरण की ध्वनि को समायोजित कर रहा है। कृपया प्रतीक्षा करें...',
            'listen_prompt': 'सुन रहा हूँ... अब बोलें।',
            'ready_prompt': 'अगले इनपुट के लिए तैयार। अब बोलें या "बाहर" कहें।'
        },
        'id-ID': {
            'name': 'Indonesian',
            'noise_prompt': 'Menyesuaikan kebisingan lingkungan. Mohon tunggu...',
            'listen_prompt': 'Mendengarkan... Berbicara sekarang.',
            'ready_prompt': 'Siap untuk input selanjutnya. Berbicara atau katakan "keluar" untuk berhenti.'
        }
    }

    def __init__(self, model_dir: Union[str, Path], language: str = 'en-US'):
        # Speech Recognition Setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Language Configuration
        self.language = language

        # Model and Tokenizer Setup
        self.model, self.tokenizer = self._load_model_and_tokenizer(model_dir)

        # Text-to-Speech Setup
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 180)  # Adjust speech rate
        self._configure_tts_language()

    def _configure_tts_language(self):
        """Configure TTS engine for different languages"""
        language_voices = {
            'en-US': 'english',
            'zh-CN': 'chinese',
            'es-ES': 'spanish',
            'fr-FR': 'french',
            'de-DE': 'german',
            'ja-JP': 'japanese',
            'ko-KR': 'korean',
            'ru-RU': 'russian',
            'ar-SA': 'arabic',
            'hi-IN': 'hindi',
            'ms-MY': 'malay',
            'id-ID': 'indonesian'
        }

        # Get available voices
        voices = self.tts_engine.getProperty('voices')

        # Try to find a voice matching the language
        for voice in voices:
            if language_voices.get(self.language, 'english').lower() in voice.languages[0].lower():
                self.tts_engine.setProperty('voice', voice.id)
                break

    def _load_model_and_tokenizer(self, model_dir: Union[str, Path]):
        # Existing implementation remains unchanged
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

        # Use MPS device if available, otherwise default to auto
        model = model.to("mps" if torch.backends.mps.is_available() else "auto")

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            trust_remote_code=True,
            encode_special_tokens=True,
            use_fast=False
        )
        return model, tokenizer

    def recognize_speech(self):
        """
        Recognize speech from microphone input in specified language.

        Returns:
            str: Transcribed text
        """
        with self.microphone as source:
            # Use language-specific prompts
            language_config = self.supported_languages.get(self.language, self.supported_languages['en-US'])

            print(language_config['noise_prompt'])
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

            print(language_config['listen_prompt'])
            audio = self.recognizer.listen(source)

        try:
            # Use language-specific recognition
            text = self.recognizer.recognize_google(audio, language=self.language)
            print(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")

        return ""

    def generate_response(self, text):
        """
        Generate model response using the loaded inference model.

        Args:
            text (str): Input text

        Returns:
            str: Model's generated response
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Traditional Chinese Medicine (TCM) expert. "
                "You should only provide TCM advice, herbal remedies, and recipes. "
                "If the question is unrelated to TCM, politely respond that you are "
                "only here to discuss TCM topics and cannot answer other questions."
                ),
            },
            {"role": "user", "content": text}
        ]

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

    def text_to_speech(self, text):
        """
        Convert text to speech using pyttsx3 in the selected language.

        Args:
            text (str): Text to convert
        """
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def run_continuous_pipeline(self):
        """
        Continuous speech recognition, processing, and text-to-speech loop.
        """
        try:
            # Use language-specific ready prompt
            language_config = self.supported_languages.get(self.language, self.supported_languages['en-US'])
            print(language_config['ready_prompt'])

            while True:
                # Speech Recognition
                input_text = self.recognize_speech()

                if not input_text:
                    continue

                # Check for exit command - use multiple language exit words
                exit_words = {
                    'en-US': ['exit', 'quit', 'bye'],
                    'zh-CN': ['退出', 'exit', 'quit'],
                    'ms-MY': ['keluar', 'berhenti', 'exit'],
                    'es-ES': ['salir', 'exit', 'quit'],
                    'fr-FR': ['sortir', 'quitter', 'exit'],
                    'de-DE': ['beenden', 'exit', 'quit'],
                    'ja-JP': ['終了', 'exit', 'quit'],
                    'ko-KR': ['종료', 'exit', 'quit'],
                    'ru-RU': ['выход', 'exit', 'quit'],
                    'ar-SA': ['خروج', 'exit', 'quit'],
                    'hi-IN': ['बाहर', 'exit', 'quit'],
                    'id-ID': ['keluar', 'berhenti', 'exit']
                }

                if input_text.lower() in exit_words.get(self.language, exit_words['en-US']):
                    print("Exiting the speech pipeline.")
                    break

                # Generate Response
                response = self.generate_response(input_text)
                print(f"AI Response: {response}")

                # Text-to-Speech
                self.text_to_speech(response)

                # Use language-specific ready prompt again
                print(language_config['ready_prompt'])

        except KeyboardInterrupt:
            print("\nPipeline stopped by user.")


@app.command()
def main(
        model_dir: Annotated[str, typer.Argument(help='Path to the model directory')],
        language: Annotated[Optional[str], typer.Option(help='Language code for speech recognition and TTS')] = 'en-US'
):
    # Updated supported languages list
    supported_languages = list(SpeechInferencePipeline.supported_languages.keys())

    if language not in supported_languages:
        raise typer.BadParameter(
            f"Unsupported language. Choose from: {', '.join(supported_languages)}")

    pipeline = SpeechInferencePipeline(model_dir, language)
    pipeline.run_continuous_pipeline()


if __name__ == '__main__':
    app()
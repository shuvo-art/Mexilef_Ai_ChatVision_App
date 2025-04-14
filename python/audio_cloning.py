import os
import sys
import noisereduce as nr
import soundfile as sf
import tempfile
import subprocess
import librosa
from elevenlabs import clone, voices, set_api_key
import requests

# Set ElevenLabs API Key
API_KEY = "sk_64d6b4479e000033f8c46242b98f1dc388a160e8855286e1"
set_api_key(API_KEY)

def convert_audio_to_wav(input_path: str) -> str:
    output_path = tempfile.mktemp(suffix=".wav")
    command = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

def delete_voice_by_id(voice_id: str):
    print(f"Deleting voice with ID: {voice_id}")
    url = f"https://api.elevenlabs.io/v1/voices/{voice_id}"
    headers = {
        "xi-api-key": API_KEY,
        "accept": "application/json"
    }
    response = requests.delete(url, headers=headers)
    if response.status_code == 204:
        print(f"Deleted voice ID: {voice_id}")
    else:
        print(f"Failed to delete voice ID: {voice_id}, Status: {response.status_code}, Message: {response.text}")

def remove_noise_and_clone_voice(input_audio_path, clone_name):
    print(f"Step 1: Starting process for: {input_audio_path}, clone_name: {clone_name}")
    description = "a person talking"

    try:
        # Step 2: Convert to WAV
        print("Step 2: Converting audio to .wav format")
        converted_wav_path = convert_audio_to_wav(input_audio_path)

        # Step 3: Load and process
        print("Step 3: Loading WAV audio with librosa")
        audio_data, sample_rate = librosa.load(converted_wav_path, sr=16000, mono=True)
        print(f"Loaded: sample_rate={sample_rate}, shape={audio_data.shape}")

        # Step 4: Reduce noise
        print("Step 4: Reducing noise")
        reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate, stationary=True, prop_decrease=0.75)

        # Step 5: Save noise-reduced to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            sf.write(tmp_file.name, reduced_noise_audio, sample_rate)
            tmp_file_path = tmp_file.name

        # Step 6: Reuse voice if already exists
        existing = [v for v in voices() if v.name == clone_name]
        if existing:
            print(f"Voice with name '{clone_name}' already exists. Reusing ID: {existing[0].voice_id}")
            os.remove(tmp_file_path)
            os.remove(converted_wav_path)
            return existing[0].voice_id

        # Step 7: Check limit and delete all old voices if full
        all_voices = voices()
        if len(all_voices) >= 30:
            print("Step 7: Voice limit reached. Deleting all previous voices...")
            for v in all_voices:
                delete_voice_by_id(v.voice_id)

        # Step 8: Clone
        print("Step 8: Cloning voice...")
        voice = clone(name=clone_name, description=description, files=[tmp_file_path])
        print(f"Voice cloned successfully, ID: {voice.voice_id}")

        os.remove(tmp_file_path)
        os.remove(converted_wav_path)
        return voice.voice_id

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    print(f"Args: {sys.argv}")
    if len(sys.argv) != 3:
        print("Usage: python audio_cloning.py <input_audio_path> <clone_name>")
        sys.exit(1)

    input_audio_path = sys.argv[1]
    clone_name = sys.argv[2]

    try:
        voice_id = remove_noise_and_clone_voice(input_audio_path, clone_name)
        print(f"Cloned voice ID: {voice_id}")
        sys.stdout.flush()
    except Exception as e:
        print(f"Execution failed: {str(e)}")
        sys.exit(1)

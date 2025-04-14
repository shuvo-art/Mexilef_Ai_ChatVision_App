import sys
import json
import openai
import os
from elevenlabs import generate, set_api_key
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch the OpenAI API key from an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    sys.exit(1)
    
# Configure ElevenLabs API key
set_api_key("sk_64d6b4479e000033f8c46242b98f1dc388a160e8855286e1")

# Unset any proxy environment variables to prevent interference
for proxy_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if proxy_var in os.environ:
        print(f"Removing proxy variable {proxy_var}: {os.environ[proxy_var]}")
        os.environ.pop(proxy_var, None)

def generate_ai_response_and_convert_to_audio(user_input, cloned_voice_id, user_data_file, output_audio_path):
    """
    Function to generate AI response based on user input and convert it to cloned voice audio.
    Supports personalized data from a JSON file with error handling.

    Parameters:
        user_input (str): The text input from the user.
        cloned_voice_id (str): The ID of the cloned voice.
        user_data_file (str): Path to the JSON file containing user and loved one personalized data.
        output_audio_path (str): The path where the generated audio will be saved.

    Returns:
        generated_audio_path (str): Path to the generated audio file, or None if an error occurs.
    """
    print(f"Step 1: Starting generate_ai_response_and_convert_to_audio")
    print(f"Input parameters - user_input: {user_input}, cloned_voice_id: {cloned_voice_id}, user_data_file: {user_data_file}, output_audio_path: {output_audio_path}")

    # Read the user data from the file
    try:
        with open(user_data_file, 'r') as f:
            user_data = json.load(f)
        print(f"Step 2: Loaded user_data from file: {user_data}")
    except FileNotFoundError as e:
        print(f"Step 2.1: Error loading user data file: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Step 2.2: Error decoding JSON from user data file: {e}")
        return None

    # Standardize field names to match the expected structure
    user_data = {
        'loved_one_name': user_data.get('lovedOneName', 'Unknown'),
        'loved_one_birthday': user_data.get('lovedOneBirthday', 'Unknown'),
        'user_birthday': user_data.get('userBirthday', 'Unknown'),
        'nickname_for_user': user_data.get('nicknameForUser', 'Unknown'),
        'favorite_song': user_data.get('favoriteSong', 'Unknown'),
        'signature_phrase': user_data.get('signaturePhrase', 'Unknown'),
        'favorite_topic': user_data.get('favoriteTopic', 'Unknown'),
        'distinct_greeting': user_data.get('distinctGreeting', 'Unknown'),
        'distinct_goodbye': user_data.get('distinctGoodbye', 'Unknown'),
        'additional_data': user_data.get('additionalData', 'Unknown')
    }
    print(f"Step 3: Standardized user_data: {user_data}")

    prompt = f"""
You are an AI assistant having a warm, caring, and supportive conversation with a user. The goal is to reply in a natural, loving tone, based on the context of the user's input, but without repeating what the user said.

Here is some information about the user:

- The user's loved one (AI's cloned voice) is named {user_data.get('loved_one_name', 'Unknown')}.
- The user and their loved one have a special bond. The loved one calls the user {user_data.get('nickname_for_user', 'Unknown')}.
- The user and their loved one have shared experiences, such as {user_data.get('favorite_song', 'Unknown')} or their favorite topics, like {user_data.get('favorite_topic', 'Unknown')}.
- The loved one has a distinctive way of greeting the user, saying: "{user_data.get('distinct_greeting', 'Unknown')}", and a special farewell phrase: "{user_data.get('distinct_goodbye', 'Unknown')}".
- The user's birthday is {user_data.get('user_birthday', 'Unknown')}, and the loved one's birthday is {user_data.get('loved_one_birthday', 'Unknown')}.
- The loved one often says: "{user_data.get('signature_phrase', 'Unknown')}".

**User input:** "{user_input}"

**Respond naturally based on the user's input.** Keep the response warm and loving, just like how the user’s loved one would respond. Don't repeat the user’s words; focus on giving a thoughtful and supportive reply. If the user brings up any of the special topics or moments, incorporate them into the conversation in a natural and caring way.

Your responses should sound like they come from the user’s loved one. Be warm, natural, and caring, incorporating relevant details about the user's loved one or their relationship only when it feels right in the conversation.

If the user mentions something related to the favorite song, signature phrase, or special moments, include that in a personal way.
"""
    print(f"Step 4: Generated prompt for OpenAI: {prompt[:100]}...")  # Truncate for brevity

    # Generate AI response using OpenAI GPT-4 with new API syntax
    try:
        print("Step 5: Calling OpenAI API to generate response")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a warm, caring, and supportive assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        ai_response_text = response.choices[0].message.content.strip()
        print(f"Step 6: OpenAI response received: {ai_response_text}")
    except Exception as e:
        print(f"Step 6.1: Error generating AI response: {str(e)} - Check OpenAI API key or network connectivity")
        # Fallback response in case OpenAI API fails
        ai_response_text = f"{user_data.get('distinct_greeting', 'Hello, dear!')} I'm here for you, {user_data.get('nickname_for_user', 'sweetheart')}. {user_data.get('signature_phrase', 'I care about you deeply.')}"
        print(f"Step 6.2: Using fallback response: {ai_response_text}")

    # Convert the AI response to speech using ElevenLabs
    try:
        print(f"Step 7: Generating audio with ElevenLabs - text: {ai_response_text[:50]}..., voice: {cloned_voice_id}")
        audio = generate(
            text=ai_response_text,
            voice=cloned_voice_id,
            model="eleven_multilingual_v2"
        )
        print("Step 8: Audio generated, writing to file")
        with open(output_audio_path, "wb") as output_file:
            output_file.write(audio)  # In version 0.2.27, generate() returns bytes directly
        print(f"Step 9: Audio successfully written to {output_audio_path}")
    except Exception as e:
        print(f"Step 9.1: Error generating speech with ElevenLabs: {str(e)} - Check ElevenLabs API key or voice ID")
        return None

    print(f"Step 10: Returning generated audio path: {output_audio_path}")
    return output_audio_path

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python generate_ai_response.py <user_input> <cloned_voice_id> <user_data_file> <output_audio_path>")
        sys.exit(1)

    user_input = sys.argv[1]
    cloned_voice_id = sys.argv[2]
    user_data_file = sys.argv[3]
    output_audio_path = sys.argv[4]

    print(f"Main: Executing with arguments - user_input: {user_input}, cloned_voice_id: {cloned_voice_id}, user_data_file: {user_data_file}, output_audio_path: {output_audio_path}")
    generated_audio_path = generate_ai_response_and_convert_to_audio(user_input, cloned_voice_id, user_data_file, output_audio_path)
    if generated_audio_path:
        print(f"Generated audio saved at: {generated_audio_path}")
    else:
        print("There was an error generating the audio.")

    # Optional: Add hardcoded example for compatibility with first version
    if user_input == "test" and cloned_voice_id == "test":
        user_data = {
            'loved_one_name': 'Mom',
            'loved_one_birthday': '1990-05-15',
            'user_birthday': '2015-11-20',
            'nickname_for_user': 'Mom',
            'favorite_song': 'You Are My Sunshine',
            'signature_phrase': 'I love you to the stars and back!',
            'favorite_topic': 'traveling',
            'distinct_greeting': 'Hey, Mom! How are you today?',
            'distinct_goodbye': 'Catch you later, love!',
            'additional_data': 'The user loves cooking and enjoys hiking in the mountains.'
        }
        user_input = "How are you my son?"
        cloned_voice_id = "etbmwBxMSr4d1abqNSYy"
        output_audio_path = "generated_audio.wav"
        try:
            print(f"Main: Running hardcoded example with user_input: {user_input}, cloned_voice_id: {cloned_voice_id}")
            generated_audio_path = generate_ai_response_and_convert_to_audio(user_input, cloned_voice_id, user_data, output_audio_path)
            if generated_audio_path:
                print(f"Generated audio (hardcoded example) saved at: {generated_audio_path}")
            else:
                print("There was an error generating the hardcoded example audio.")
        except Exception as e:
            print(f"Main: Hardcoded example failed with error: {str(e)}")
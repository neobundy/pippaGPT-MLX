# PippaGPT MLX - Personalized, Ingesting, Persistent, Polymorphic, Adaptive GPT Chatbot

![Pippa Logo](images/ai.png)

Pippa is an open-source Large Language Model (LLM) chatbot project based on the LangChain framework. It aims to offer a personalized and adaptive conversational experience.

Pippa incorporates various features to enhance the adaptation process, including the ability to customize the chatbot's personality, ingest documents for learning, 
remember conversation history, switch between different characters, and adapt to the user's needs. 

MLX is a new feature that allows you to use Apple MLX framework. It's still experimental and not fully integrated yet.

## Features

* Personalized: Users can tailor the chatbot's personality for a more engaging and customized conversational experience. 
* Ingesting: Pippa can ingest documents and extract context, allowing users to incorporate their own documents into the chatbot's knowledge base. 
* Persistent: Pippa retains the history of conversations, enabling users to revisit past conversation contexts. 
* Polymorphic: Pippa can morph into characters of your choice, easily customizable through custom instructions. 
* Adaptive: Pippa adjusts to users' needs by modifying its conversational style and responses based on context.

Retrieval QA module was inspired by [localGPT by Prompt Engineering](https://github.com/PromtEngineer/localGPT).

* Selected MLX LLM models are supported including MLX whisper models.
* OpenAI specific features are not supported when using MLX models.
* MLX models retain no context. It's not compatible with LangChain memory types.
* MLX features require the following package:

```bash
  pip install mlx mlx-llm
```
Or simply rerun the following command:

```bash
   pip install -r requirements.txt
```


## Project Background

The Pippa project started as a personal hobby to explore the possibility of creating an AI daughter character. 
Though still in the early stages, the project is regularly updated with new features and improvements. 
The default roles in Pippa are "System" for system messages, "Pippa" as the caring daughter character, and "Bundy" as her father.

When using TTS/STT engines, Pippa uses the ElevenLabs API. You can sign up for a free account: https://elevenlabs.io.

For a better experience, assign unique voices to the characters.

## Installation

Tested with Python 3.11.

To install Pippa, follow these steps:

1. Clone the repository:

```bash
   git clone https://github.com/neobundy/pippaGPT-MLX.git
```

2. Navigate to the project directory: 

```bash
   cd pippaGPT-MLX
```

3. Install the required dependencies: 

```bash
   pip install -r requirements.txt
```

> **Note:** If you encounter errors, you may need to install additional dependencies like `ffmpeg` and `portaudio`. On macOS, you can use Homebrew.
>
> To install them, run the following commands:
>
> ```bash
> brew install ffmpeg
> brew install portaudio
> ```

4. Create or copy `env_sample` to `.env` file in the root folder of the project and add your API keys: 

Note that HUGGING_FACE_API_KEY is for future use.

```bash
OPENAI_API_KEY=
XI_API_KEY=
HUGGING_FACE_API_KEY=
SERPAPI_API_KEY=
```   
   
 Alternatively, you can export these environment variables in your terminal.

5. copy `characters_samply.py` to `characters.py` and edit the file to customize your AI's name and personality.

6. Copy `settings_private_sample.py` to `settings_private.py`. `settings_sample.py` to `settings.py` and edit the files to customize your settings.

7. Choose LLMs model in `settings.py` file:

```python
DEFAULT_GPT_MODEL = "gpt-3.5-turbo"
DEFAULT_GPT_HELPER_MODEL = "gpt-3.5-turbo-16k"
DEFAULT_GPT_QA_HELPER_MODEL = "gpt-3.5-turbo-16k"
``` 

* DEFAULT_GPT_MODEL - Main model for conversation.
* DEFAULT_GPT_HELPER_MODEL - Model for summarization buffer memories.
* DEFAULT_GPT_QA_HELPER_MODEL - Model for retrieval QA.

Large context needs more tokens. 16k tokens is enough for most cases.

GPT-4 model for DEFAULT_GPT_MODEL is highly recommended for better experience, but note that it's 10x expensive and only available for pre-paid OpenAI accounts. 

8. Some sensitive or user-specific settings found in the `settings_private.py` such as Zep vector store server or audio server URLs.

```python
ZEP_API_URL = "http://localhost:8000"
DEFAULT_MEMORY_TYPE = "Summary Buffer"
AUDIO_SERVER_URL = "http://localhost:5000"
```

## Memories

By default, LangChain's "Summary Buffer" memory is used to retain the conversation context.

Pippa supports six types of memories:

1. Sliding Window: ConversationBufferWindowMemory - retains a specified number of messages.
2. Token Buffer: ConversationTokenBufferMemory - retains messages based on a given number of tokens.
3. Summary Buffer: ConversationSummaryBufferMemory - retains a summarized history while also storing all messages.
4. Summary: ConversationSummaryMemory - retains only the summary.
5. Buffer: ConversationBufferMemory - the most basic memory type that stores the entire history of messages as they are.
6. Zep: vector store

Zep is highly recommended for large context. It can be run locally as a Docker container. Edit the `settings_private.py`.

Summaries appear when Summary type memory is selected including Zep. Summaries are generated by the summarization GTP helper model.

Note that it takes a while for Zep to index and generate summaries. When not ready, "Summarizing...please be patient." message will appear.

```python
ZEP_API_URL="http://localhost:8000"
```

Zep server can be run on any host or port. If you run it on a different host, make sure to update the `ZEP_API_URL` variable in `settings_private.py`.

Visit https://www.getzep.com/ to learn how to run Zep.

## Running the App

To run the Pippa app, use the following command:

```bash
streamlit run main.py
```

The app will automatically start the audio server which listens on port 5000. If needed, you can manually run the audio server by executing `audio_server.py`.

## Ingesting Your Documents

To ingest your own documents for Pippa to learn from, follow these steps:

1. Place your documents (e.g., PDF, DOCX, XLSX, TXT, MD, PY) in the `docs` folder.  
2. Run the `vectordb.py` script to create your vector database: 

```bash
   python vectordb.py
```

## TTS/STT Engines - ElevenLabs

If you have an ElevenLabs API key, you can use their TTS(Text-to-Speech) engine with Pippa. 

STT(Speech-to-Text) is handled by OpenAI's Whisper-1 model or MLX Whisper depending on your choice in WebUI.

Follow these steps:

1. Run the `tts.py` script to get the available voice names and IDs from your ElevenLabs account: 

```bash
   python tts.py
```

Update the following variables in `settings_private.py` with the appropriate voice IDs: 

```python
VOICE_ID_AI = ""
VOICE_ID_SYSTEM = ""
VOICE_ID_HUMAN = ""
```

2. The TTS/STT features are supported as a Flask audio server. The server will automatically run and listen on port 5000 when the app is started. 
You can also manually run the server by executing `audio_server.py`.
3. When Use Audio checkbox is checked, the app will use the audio server to convert text to speech and speech to text. On Mac, you should allow the app to use Microphone in System Preferences. On first use, you'll be prompted to allow the app to use the microphone.
4. Record button starts recording, click Stop when done. The app will automatically convert the recorded audio to text and send it to the chatbot.
5. To TTS feature, click Speak button assigned to any message. The app will automatically convert the text to speech and play the audio.
6. To go back to typing mode, just uncheck the Use Audio checkbox.

## Prompt Keywords

You can customize the prompt keyword prefixes used in Pippa by editing the `settings.py` file:  

* `PROMPT_KEYWORD_PREFIX_SYSTEM`: Used for temporary system messages (default: "system:") 
* `PROMPT_KEYWORD_PREFIX_CI`: Used for replacing custom instructions (default: "ci:") 
* `PROMPT_KEYWORD_PREFIX_QA`: Used for retrieval QA based on your documents in the `docs` folder (default: "qa:")
* `PROMPT_KEYWORD_PREFIX_GOOGLE`: Used for searching the web for given information (default: "google:")
* `PROMPT_KEYWORD_PREFIX_WIKI`: Used for searching Wikipedia (default: "wiki:")
* `PROMPT_KEYWORD_PREFIX_MATH`: Used for mathematical query (default: "math:")
* `PROMPT_KEYWORD_PREFIX_MIDJOURNEY`: Used for generating Midjourney prompts (default: "midjourney:")

## How Conversations and Context Windows are managed and saved

* Taking a Snapshot: A snapshot is captured whenever Pippa responds. This snapshot includes the entire conversation, not just the context window.
* Saving Conversations: The conversation is saved as a JSON file in the conversations folder when the user clicks the "Export Conversation" button.
* Last User Input: The most recent user input is saved in the last_user_input.md file within the temp folder. This is utilized to restore the last input in the event of a critical app error that resets the input.
* Starting a New Conversation: When initiating a new conversation, both the context window and the conversation are reset. If you choose a previous conversation from the "Load Conversation" dropdown menu, the JSON file is loaded and the context window is restored. To restore the entire conversation, however, you'll need to rerun the app. Note that Zep is not compatible with other LangChain memory types; if you switch to Zep during an ongoing conversation, make sure to load the latest snapshot to restore both the context and the entire conversation into Zep's vector store.

## Streaming and Costs Information

Streaming is enabled by default. To disable it, modify the settings.py file as follows:

```python
STREAMING_ENABLED = True
```

When streaming is enabled, the costs are approximations based on OpenAI's documentation. To obtain exact costs, you'll need to disable streaming.

Note that the cost calculation does not include other expenses incurred by auxiliary GPT models, such as those for summarization and QA.

## Agents

Set the following constant in `settings.py`:

```python
DEFAULT_GPT_AGENT_HELPER_MODEL = "gpt-4"
```

‼️Warning: This operation is very expensive in terms of OpenAI tokens.

```python
MAX_AGENTS_ITERATIONS = 8
```

The number of iterations determines how many times the agent will run. A higher number of iterations generally leads to more accurate answers, but it also consumes more tokens.  

Please note that the Google Search agent may need to perform multiple queries to obtain the best answer. 

For instance, if you ask "Who is the oldest among the heads of state of South Korea, the US, and Japan?", the agent will likely need to query at least 3-4 times to obtain the final answer.

The same model may respond differently to the same query. Even 'gpt-4' doesn't always perform the best, but highly recommended. Experiment with different models.  

Note that even at the LangChain level, it's highly experimental. It may not work as expected.

### Search Web

It's a hit-or-miss situation depending on your prompting skills. You need a SerpApi API key to use the Google search feature: https://serpapi.com. The provided final answer serves as an intermediate prompt for the main model.

### Search Wikipedia

The agent first attempts to find the relevant Wikipedia page for the given query. If found, it will return the summary of the page and search for the specific term within the summary.

### Math

LLMs are not known for their proficiency in math. The math agent provides accurate answers for highly complex math problems.

## Managing Vector DB

To manage the vector database, run the `vectordb.py` script. 

```bash
   python vectordb.py
```

You have the following options:

* (C)reate DB: Create a new vector database in the `settings.CHROMA_DB_FOLDER` folder with a collection named `settings.VECTORDB_COLLECTION`.
* (E)mbed conversations: Embed conversations from the `settings.CONVERSATION_SAVE_FOLDER` folder into the vector database to serve as long-term memory.
* (D)elete collection: Delete the vector database collection. This action will not delete the vector database itself.
* (Q)uery the DB: Query the vector database in a loop. Enter 'exit' or 'quit' to exit the loop.

Note that when you choose the (E)mbed conversations option, only the existing exported conversations `*.json` will be embedded into the vector database, excluding `snapshot.json`.

## Troubleshooting

If you encounter errors when running the app, try the following steps:

```bash

pip install --upgrade charset_normalizer
pip install --upgrade openai
pip install --upgrade langchain

```

## License

Pippa is released under the MIT license. Feel free to use, modify, and distribute the code for personal or commercial purposes.
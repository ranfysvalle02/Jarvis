import base64
import logging
import os
import random
import time

import openai
import streamlit as st
from audiorecorder import audiorecorder
from bot import JarvisBot
from constants import AUDIO_INPUT, AUDIO_OUTPUT

openai.api_key = os.getenv("OPENAI_API_KEY")
widget_id = (id for id in range(1, 10000))

logging.basicConfig(
    filename="app.log",
    filemode="a",
    format="%(asctime)s.%(msecs)04d %(levelname)s {%(module)s} [%(funcName)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )


@st.cache_resource
def get_agent():
    logger.info("Loading RAG Bot ...")
    return JarvisBot(logger, st, stream=True)


st.markdown("----")

agent = get_agent()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# audio = audiorecorder(
#     start_prompt="Speaking",
#     stop_prompt="Finish Speaking",
#     pause_prompt="Pause",
#     key=next(widget_id),
# )


# Accept user input
if prompt := st.chat_input(placeholder="What's up"):
    # Hear Audio Input
    # if len(audio) > 0:
    #     audio.export(AUDIO_INPUT, format="mp3")
    #     audio.empty()
    #     agent.listen_input(AUDIO_INPUT)
    # Read Text Input
    agent.read_input(prompt)

    text_input = agent.parse_input_to_text()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": text_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(text_input)

    # Generate response from bot
    response = agent.respond(text_input, AUDIO_OUTPUT)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if response.audio_file:
            autoplay_audio(AUDIO_OUTPUT)

        if response.stream:
            for chunk in response.stream:
                if isinstance(chunk, str):
                    full_response += chunk
                    time.sleep(0.05)
                elif chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content + " "

                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")

        elif response.text:
            assistant_response = response.text

            full_response += assistant_response + " "

            # Simulate stream of response with milliseconds delay
            # for chunk in assistant_response.split():
            #     full_response += chunk + " "
            #     time.sleep(0.05)
            #     # Add a blinking cursor to simulate typing
            #     message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})


# add a default value to the chat input
# default_chat_input_value = " "
# js = f"""
#     <script>
#         function insertText(dummy_var_to_force_repeat_execution) {{
#             var chatInput = parent.document.querySelector('textarea[data-testid="stChatInput"]');
#             var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
#             nativeInputValueSetter.call(chatInput, "{default_chat_input_value}");
#             var event = new Event('input', {{ bubbles: true}});
#             chatInput.dispatchEvent(event);
#         }}
#         insertText({len(st.session_state.messages)});
#     </script>
#     """
# st.components.v1.html(js)

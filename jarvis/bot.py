import itertools
import os
import uuid
from collections.abc import Iterable
from typing import List, Optional

import openai
import requests
from actionweaver import action
from actionweaver.llms.azure.chat import ChatCompletion
from actionweaver.llms.openai.tokens import TokenUsageTracker
from bs4 import BeautifulSoup
from constants import AUDIO_INPUT, AUDIO_OUTPUT
from openai import OpenAI

from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings import GPT4AllEmbeddings
from langchain.document_loaders import PlaywrightURLLoader

openai.api_key = ""
openai.api_version = "2023-10-01-preview"
openai.api_type = "azure"

import pymongo 

MONGODB_URI = ""  
DATABASE_NAME = ""  
COLLECTION_NAME = ""

from dataclasses import dataclass


@dataclass
class BotResponse:
    text: Optional[str] = None
    stream: Optional[itertools._tee] = None
    audio_file: Optional[str] = None


class JarvisBot:
    def __init__(
        self,
        logger,
        st,
        stream=False,
        audio_output=AUDIO_OUTPUT,
        system_str="You are a helpful assistant. Please do not try to answer the question directly.",
    ):
        self.logger = logger
        self.st = st
        self.token_tracker = TokenUsageTracker(budget=3000, logger=logger)
        self.llm = ChatCompletion(
            model="gpt-4", azure_deployment="gpt-4",
            azure_endpoint="https://.openai.azure.com/", api_key="",
            api_version="2023-10-01-preview", 
            token_usage_tracker = TokenUsageTracker(budget=2000, logger=logger), 
            logger=logger)

        self.mongodb_client = pymongo.MongoClient(MONGODB_URI)
        self.gpt4all_embd = GPT4AllEmbeddings()
        self.client = pymongo.MongoClient(MONGODB_URI)
        self.db = self.client[DATABASE_NAME]  
        self.collection = self.db[COLLECTION_NAME]  
        self.vectorstore = MongoDBAtlasVectorSearch(self.collection, self.gpt4all_embd)  
        self.index = self.vectorstore.from_documents([], self.gpt4all_embd, collection=self.collection)

        self.audio_input = None
        self.text_input = None
        self.audio_output = audio_output
        self.stream = stream

        self.messages = [{"role": "system", "content": system_str}]

    def transcribe_audio(self, audio_file):
        client = OpenAI()
        audio_file = open(audio_file, "rb")
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
        return transcript.text

    @action("Speak", stop=True)
    def speak(self, text):
        """
        Give speech from text.

        Args:
            text (str): The text to read.
        """
        response = openai.audio.speech.create(
            model="tts-1", voice="shimmer", input=text
        )
        response.stream_to_file(self.audio_output)
        return BotResponse(audio_file=self.audio_output, stream=text)

    def listen_input(self, audio_input):
        self.audio_input = audio_input

    def read_input(self, text):
        self.text_input = text

    def parse_input_to_text(self):
        input = ""
        if self.audio_input is not None:
            input += f"{self.transcribe_audio(self.audio_input)} \n"

        if self.text_input is not None:
            input += f"{self.text_input}"

        return input

    def respond(self, input, audio_output=None, speak=True):
        if audio_output is not None:
            self.audio_output = audio_output

        response = self.__call__(input)

        if isinstance(response, str):
            response = BotResponse(text=response)
        elif isinstance(response, Iterable):
            response = BotResponse(stream=response)

        self.audio_input = None
        self.text_input = None

        return response

    def __call__(self, query):
        self.messages += [{"role": "user", "content": query}]

        response = self.llm.create(
            self.messages,
            stream=self.stream,
            actions=[self.learn, self.speak, self.answer_question],
        )

        return response

    def recall(self, text):
        response = self.index.similarity_search_with_score(text) 
        str_response = []
        for vs in response:
            score = vs[1]
            v = vs[0]
            print("URL"+v.metadata["source"]+";"+str(score))
            str_response.append({"URL":v.metadata["source"],"content":v.page_content[:800]})
        
        if len(str_response)>0:
            return f"VectorStore Search Results (source=URL):\n{str_response}"[:5000]
        else:
            return "No information on that topic"

    @action(name="AnswerQuestion", stop=True)
    def answer_question(self, query: str):
        """
        Invoke this method to answer a question. e.g. what is ActionWeaver?

        Parameters
        ----------
        query : str
            The query to be used for answering a question.
        """

        context_str = self.recall(query)
        context = (
            "Context:\n"
            f"{context_str}\n"
            "---\n"
            f"User: {query}\n"
            "Are you able to answer the user's question based on the context above?\n"
            "If yes, please answer the question. If no, performs a Google search instead. Your Response:"
        )

        return self.llm.create(
            [
                {"role": "user", "content": context},
            ],
            actions=[self.search],
        )

    @action(name="GoogleSearch", stop=True)
    def search(self, query: str):
        """
        Perform a Google search and return query results with titles and links.

        Parameters
        ----------
        query : str
            The search query to be used for the Google search.

        Returns
        -------
        str
            A formatted string containing Google search results with titles, snippets, and links.
        """

        with self.st.spinner(f"Searching '{query}'..."):
            from langchain.utilities import GoogleSearchAPIWrapper

            search = GoogleSearchAPIWrapper()
            res = search.results(query, 10)
            formatted_data = ""

            # Iterate through the data and append each item to the formatted_data string
            for idx, item in enumerate(res):
                formatted_data += f"({idx}) {item['title']}: {item['snippet']}\n"
                formatted_data += f"[Source]: {item['link']}\n\n"

        return f"Here are Google search results:\n\n{formatted_data}"

    @action("Learn", stop=True)
    def learn(self, text: str = "", urls: List[str] = []):
        """
        Learn and read content from the provided sources and remember them.

        Parameters
        ----------
        text : str
            Text to be read.
        urls : List[str]
            List of URLs to scrape.
        """

        def chunk(text):
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            text_splitter = RecursiveCharacterTextSplitter(
                # Set a really small chunk size, just to show.
                chunk_size=4000,
                chunk_overlap=200,
                length_function=len,
                add_start_index=True,
            )

            buf = text_splitter.create_documents([text])

            return {
                "documents": [doc.page_content for doc in buf],
                "ids": [uuid.uuid4().hex for _ in range(len(buf))],
            }

        if len(text) > 0:
            with self.st.spinner(f"Learning.."):
                self.index.add_documents(
                        **chunk(text)
                ) 

        if len(urls) > 0:
            with self.st.spinner(f"Learning the content in {urls}"):
                for url in urls:
                    # Retrieve the HTML content from the URL
                    response = requests.get(url)

                    if response.status_code == 200:
                        html_content = response.text

                        # Parse the HTML content
                        soup = BeautifulSoup(html_content, "html.parser")

                        # Find all <p> tags and extract their text
                        paragraphs = soup.find_all("p")

                        # Extract and print the text from the <p> tags
                        for paragraph in paragraphs:
                            if len(paragraph.text) > 0:
                                self.index.add_documents(**chunk(paragraph.text))

        return f"Content has been learned."


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        filename="bot.log",
        filemode="a",
        format="%(asctime)s.%(msecs)04d %(levelname)s {%(module)s} [%(funcName)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    agent = JarvisBot(logger, None)

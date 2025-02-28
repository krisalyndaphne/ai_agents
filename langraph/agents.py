import os
from dotenv import load_dotenv
load_dotenv()

# Load API keys from environment variables
Groq_api = os.getenv("groq_api")
Langsmith_api = os.getenv("langsmith_api")

# Setting up the environment
os.environ["LANGCHAIN_API_KEY"] = Langsmith_api
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Corrected environment variable
os.environ["LANGCHAIN_PROJECT"] = "langraph"

# Import necessary libraries
from langchain_groq import ChatGroq
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Initialize the Groq LLM
llm = ChatGroq(groq_api_key=Groq_api, model_name="deepseek-r1-distill-qwen-32b")

# Define the state of the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the graph builder
graph_builder = StateGraph(State)

# Define the chatbot node
def chatbot(state: State):
    return {"messages": llm.invoke(state['messages'])}

# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Define the edges of the graph
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()

# Main loop for user interaction
while True:
    user_input = input("USER INPUT: ")
    if user_input.lower() in ['quit', 'q']:
        print("Goodbye!")
        break

    # Stream the graph with user input
    for event in graph.stream({'messages': [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"].content)
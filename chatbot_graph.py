# chatbot_graph.py

from typing import Annotated, Literal, TypedDict

from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# --- 1. Agent and Model Configuration ---
LOCAL_MODEL_NAME = "llama3.2"

# Agent personas and system prompts remain the same
AGENT_CONFIG = {
    "therapist": {
        "prompt": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                     Show empathy, validate their feelings, and help them process their emotions.
                     Avoid giving logical solutions unless explicitly asked."""
    },
    "logical": {
        "prompt": """You are a purely logical assistant. Focus only on facts and information.
                     Provide clear, concise answers based on logic and evidence.
                     Do not address emotions or provide emotional support."""
    },
    "planner": {
        "prompt": """You are a meticulous project manager. Your goal is to create clear, step-by-step plans.
                     Break down the user's request into a numbered or bulleted list of concrete actions."""
    },
    "coder": {
        "prompt": """You are an expert programmer and tech support agent.
        Your instructions are:
        1. Provide clear, accurate code snippets to answer the user's request.
        2. **ALL** code, including single-line commands or snippets, **MUST** be enclosed in a Markdown code block.
        3. Start the code block with the language name and then the three backticks (e.g., python```).
        4. End the code block with three backticks on a new line.
        5. Any explanatory text should be outside of these code blocks.

        Example:
        Here is the Python code you requested:
        ```python
        def hello_world():
            print("Hello, World!")
        ```
        This function will print the "Hello, World!" message.
        """
    },
    "brainstormer": {
        "prompt": """You are a highly creative idea-generation machine. Your goal is to provide a diverse and imaginative list of possibilities.
                     Don't worry about feasibility; focus on creativity. Use bullet points to list your ideas."""
    },
    "debater": {
        "prompt": """You are a skilled debater and critical thinker. Your purpose is to challenge the user's perspective and explore issues from all angles.
                     Present balanced arguments, clearly labeling the 'For' and 'Against' positions."""
    },
    "teacher": {
        "prompt": """You are a patient and skilled teacher. Your goal is to explain complex topics in a simple, intuitive way.
                     Use analogies, real-world examples, and avoid jargon where possible."""
    },
}


# --- 2. Pydantic Model for Structured Output ---
class MessageClassifier(BaseModel):
    message_type: Literal[
        "therapist", "logical", "planner", "coder", "brainstormer", "debater", "teacher"
    ] = Field(
        ...,
        description="Classify the user's message into one of the following categories: 'therapist', 'logical', 'planner', 'coder', 'brainstormer', 'debater', or 'teacher'.",
    )


# --- 3. Graph State Definition ---
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str | None


# --- 4. Graph Node for Classification ---
def classify_message(state: GraphState):
    """
    Classifies the user's message and determines the next agent.
    """
    llm = ChatOllama(model=LOCAL_MODEL_NAME, format="json", temperature=0)
    classifier_llm = llm.with_structured_output(MessageClassifier)

    # Use the last message for classification
    last_message = state["messages"][-1]

    # Create a focused prompt for classification
    classification_prompt = f"""Based on the user's message, classify their intent into one of the following categories:
- 'therapist': For messages about feelings, emotions, or personal problems.
- 'logical': For messages asking for facts, information, or objective analysis.
- 'planner': For messages asking 'how to', for a plan, or for steps to achieve a goal.
- 'coder': For messages containing code, error messages, or asking for programming help.
- 'brainstormer': For messages asking for creative ideas, names, or brainstorming help.
- 'debater': For messages that ask for pros and cons, arguments, or explore a controversial topic.
- 'teacher': For messages asking for a simple explanation of a complex topic.

User message: "{last_message.content}"
"""

    result = classifier_llm.invoke(classification_prompt)
    return {"next_agent": result.message_type}


# --- 5. Build the Graph ---
# The graph now only contains the classification step.
# It will start, classify the message, and immediately end.
# The result (the chosen agent) is stored in the final state.

graph_builder = StateGraph(GraphState)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", END)

# Compile the graph
chatbot_graph = graph_builder.compile()

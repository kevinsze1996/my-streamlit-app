# app.py

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama.chat_models import ChatOllama

# Import the graph and agent config from our other file
from chatbot_graph import chatbot_graph, AGENT_CONFIG, LOCAL_MODEL_NAME

st.set_page_config(page_title="Multi-Agent Chatbot", page_icon="🤖")

st.title("Multi-Agent Chatbot")
st.caption(
    "A versatile chatbot to help you with various tasks using different agent personas."
)

# --- Configuration ---
ASSISTANT_AVATAR = (
    "assets/chatbot_avatar.jpg"  # Make sure this image is in the same folder
)


# --- Core Streaming Logic ---


def get_llm_response(user_input, chat_history):
    """
    This function gets the appropriate agent, generates a response from the LLM,
    and streams it back to the UI in real-time, now with a fix for extra code blocks.
    """
    # 1. Get agent and system prompt
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    final_state = chatbot_graph.invoke(initial_state)
    agent_name = final_state.get("next_agent", "logical")

    if st.session_state.get("show_routing", False):
        st.sidebar.info(f"Routed to: **{agent_name.title()}** Agent")

    system_prompt = AGENT_CONFIG[agent_name]["prompt"]

    # 2. Prepare messages for the LLM
    graph_messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            graph_messages.append(HumanMessage(content=msg["content"]))
        else:
            graph_messages.append(AIMessage(content=msg["content"]))

    llm_messages = [SystemMessage(content=system_prompt)] + graph_messages

    # 3. Get the stream from the LLM
    llm = ChatOllama(model=LOCAL_MODEL_NAME, temperature=0.7)
    llm_stream = llm.stream(llm_messages)

    # 4. Process the stream in real-time with robust state management
    in_code_block = False
    ignore_next_fence = False  # Flag to prevent creating empty blocks

    for chunk in llm_stream:
        chunk_content = chunk.content

        # If we just exited a block, ignore chunks that are only whitespace
        if ignore_next_fence and not chunk_content.strip():
            continue

        while "```" in chunk_content:
            parts = chunk_content.split("```", 1)
            part_before = parts[0]
            chunk_content = parts[1]

            # If we encounter actual text, the next fence is not spurious
            if part_before.strip():
                ignore_next_fence = False

            yield part_before

            # --- State Transition ---
            if not in_code_block:
                # Trying to ENTER a code block
                if ignore_next_fence:
                    ignore_next_fence = False  # Consume the flag and ignore the fence
                else:
                    yield "\n\n```"
                    in_code_block = True
            else:
                # Trying to EXIT a code block
                yield "```\n\n"
                in_code_block = False
                ignore_next_fence = True  # Set flag to ignore potential phantom fences

        # Yield the remainder of the chunk
        if chunk_content:
            if chunk_content.strip():
                ignore_next_fence = False
            yield chunk_content

    # After the loop, if the block is still open, close it
    if in_code_block:
        yield "\n```\n"


# --- Streamlit App ---

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.session_state.show_routing = st.checkbox("Show Agent Routing", value=True)

# Initialize and manage chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not st.session_state.chat_history:
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": "Hello! How can I help you today?",
        }
    )

# Display past messages
for msg in st.session_state.chat_history:
    try:
        avatar = ASSISTANT_AVATAR if msg["role"] == "assistant" else "user"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True)
    except FileNotFoundError:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

# Get and process user input
if prompt := st.chat_input("Please enter your message here..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            response = st.write_stream(
                get_llm_response(prompt, st.session_state.chat_history)
            )
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    except Exception as e:
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            error_message = f"Sorry, I encountered an error: {str(e)}"
            st.error(error_message)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": error_message}
            )

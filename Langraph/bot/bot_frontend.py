import streamlit as st
from langchain_core.messages import HumanMessage
from bot_backend import chatBot

CONFIG = {"configurable": {"thread_id": "thread-1"}}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("type here")

if user_input:
    
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # res = chatBot.stream(
    #     {"messages": [HumanMessage(content=user_input)]},
    #     config=CONFIG
    # )

    # ai_msg = res["messages"][-1].content



    with st.chat_message("assistant"):
        ai_msg=st.write_stream(
            message_chunk.content for message_chunk, metadata in chatBot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable":{"thread_id":"thread-1"}},
                stream_mode="messages"
            )
        )
        
        
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_msg
    })
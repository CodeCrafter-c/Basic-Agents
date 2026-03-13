import streamlit as st
from langchain_core.messages import HumanMessage
from bot_backend import chatBot
import uuid


# utility functions

def genrate_thread_id():
    thread_id=uuid.uuid4()
    return str(thread_id)

def newChat():
    thread_id=genrate_thread_id()
    st.session_state["thread_id"]=thread_id
    st.session_state["message_history"]=[]
    # add_chats(st.session_state["thread_id"])

def add_chats(thread_id,title):
    if thread_id not in st.session_state["previous_chats"]:
        st.session_state["previous_chats"][thread_id]=title


def show_chats():
    chats = list(st.session_state["previous_chats"].items())[::-1]

    for thread_id, title in chats:
        if st.sidebar.button(title):
            st.session_state["thread_id"] = thread_id
            load_conversations(thread_id)
            
            
            
def load_conversations(thread_id):
    state=chatBot.get_state(config={"configurable":{"thread_id":thread_id}})
    messages= state.values.get('messages',[])
    history=[]
    for msg in messages:
        if(isinstance(msg,HumanMessage)):
            history.append({
                'role':'user',
                'content':msg.content
            })
        else:
            history.append({
                'role':'assistant',
                'content':msg.content
            })
        st.session_state["message_history"]=history



if "previous_chats" not in st.session_state:
    st.session_state["previous_chats"]={}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if 'thread_id' not in st.session_state:
    st.session_state["thread_id"]=genrate_thread_id()



# -----------------------------------------------ui
st.sidebar.title("let's chat")

if st.sidebar.button("new chat"):
    newChat()

st.sidebar.header("My chats")

show_chats()




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

    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    with st.chat_message("assistant"):
        ai_msg=st.write_stream(
            message_chunk.content for message_chunk, metadata in chatBot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
            )
        )
        
    state = chatBot.get_state(config)
    print(state)
    title = state.values.get("title")    
    add_chats(st.session_state["thread_id"],title)
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_msg
    })
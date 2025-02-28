import os
import streamlit as st # type: ignore
from dotenv import load_dotenv # type: ignore
from langchain.chat_models import ChatOpenAI # type: ignore
from langchain.schema import HumanMessage # type: ignore
from langchain.agents import AgentType, initialize_agent, load_tools # type: ignore
from langchain.callbacks import StreamlitCallbackHandler # type: ignore
from langchain.memory import ConversationBufferMemory # type: ignore
from langchain.prompts import MessagesPlaceholder # type: ignore

load_dotenv()

def create_agent_chain():
     chat = ChatOpenAI(
         model_name=os.environ["OPENAI_API_MODEL"],
         temperature=os.environ["OPENAI_API_TEMPERATURE"],
         streaming=True,
     )
     # OpenAI Functions AgentのプロンプトにMemoryの会話履歴を追加するための設定
     agent_kwargs = {
         "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
     }
     # OpenAI Functions Agentが使える設定でMemoryを初期化
     memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

     tools = load_tools(["ddg-search", "wikipedia"])
     return initialize_agent(
          tools, 
          chat, 
          agent=AgentType.OPENAI_FUNCTIONS,
          agent_kwargs=agent_kwargs,
          memory=memory,
     )

st.title("langchain-streamlit-app")

if "agent_chain" not in st.session_state:
     st.session_state.agent_chain = create_agent_chain()


if "messages" not in st.session_state: # st.session_stateにmessagesがない場合
     st.session_state.messages = []      # st.session_state.messagesを空のリストで初期化

for message in st.session_state.messages:   # st.session_state.messagesでループ
    with st.chat_message(message["role"]):  # ロールごとに
         st.markdown(message["content"])     # 保存されているテキストを表示

prompt = st.chat_input("What is up?")
print(prompt)

if prompt:  # 入力された文字列がある（Noneでも空文字列でもない）場合
    # ユーザーの入力内容をst.session_state.messagesに追加
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):  # ユーザーのアイコンで
         st.markdown(prompt)        # promptをマークダウンとして整形して表示

    with st.chat_message("assistant"):  # AIのアイコンで
         callback = StreamlitCallbackHandler(st.container())
         agent_chain = create_agent_chain()
         response = st.session_state.agent_chain.run(prompt, callbacks=[callback])
         st.markdown(response)

    # 応答をst.session_state.messagesに追加
    st.session_state.messages.append({"role": "assistant", "content": response})

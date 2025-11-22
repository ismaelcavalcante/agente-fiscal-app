from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

def convert_history_to_lc(history):
    """
    Se a mensagem já é LangChain Message → retorna como está.
    Se for dict → converte para LangChain Message.
    """
    lc_messages = []
    for msg in history:
        if isinstance(msg, BaseMessage):
            lc_messages.append(msg)
        else:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append(AIMessage(content=content))
    return lc_messages


def lc_to_streamlit(msg: BaseMessage):
    return {
        "role": "assistant" if isinstance(msg, AIMessage) else "user",
        "content": msg.content
    }
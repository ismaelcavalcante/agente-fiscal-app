from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage


def dict_to_lc(msg: dict) -> BaseMessage:
    """
    Converts a Streamlit message dict (role/content) into a LangChain message.
    Ensures uniformity across the application.
    """
    role = msg.get("role")
    content = msg.get("content", "")

    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    else:
        # fallback to human message
        return HumanMessage(content=content)


def lc_to_dict(msg: BaseMessage) -> dict:
    """
    Converts LangChain messages back to Streamlit dict format.
    """
    role = msg.type.replace("human", "user").replace("ai", "assistant")
    return {"role": role, "content": msg.content}


def convert_history_to_lc(history: list[dict]) -> list[BaseMessage]:
    """
    Converts Streamlit session_state messages into LangChain-compatible list.
    """
    return [dict_to_lc(m) for m in history]


def convert_history_to_streamlit(messages: list[BaseMessage]) -> list[dict]:
    """
    Converts LangChain/LLM messages back into Streamlit dict format.
    """
    return [lc_to_dict(m) for m in messages]
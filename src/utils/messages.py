from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

def sanitize_messages(history):
    fixed = []
    for msg in history:
        if isinstance(msg, BaseMessage):
            fixed.append(msg)
            continue
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "assistant":
                fixed.append(AIMessage(content=content))
            else:
                fixed.append(HumanMessage(content=content))
    return fixed
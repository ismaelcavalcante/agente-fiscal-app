from langchain_core.messages import AIMessage

class FakeGraph:
    def __init__(self, answer="OK"):
        self.answer = answer

    def invoke(self, state, config=None):
        messages = state.get("messages", [])
        messages = list(messages)
        messages.append(AIMessage(content=self.answer))
        return {"messages": messages}
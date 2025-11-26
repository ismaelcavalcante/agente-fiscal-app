class FakeSession:
    def __init__(self):
        self.state = {}

    def __getitem__(self, key):
        return self.state.get(key)

    def __setitem__(self, key, value):
        self.state[key] = value

    def __contains__(self, key):
        return key in self.state


class FakeChatMessage:
    def __init__(self, role):
        self.role = role

    def write(self, text):
        # Apenas registra para debug
        pass


class FakeStreamlit:
    def __init__(self):
        self.session_state = FakeSession()

    def chat_message(self, role):
        return FakeChatMessage(role)

    def chat_input(self, _):
        return None  # comportamento definido no teste

    def warning(self, msg):
        pass

    def error(self, msg):
        pass

    def stop(self):
        raise RuntimeError("Streamlit stop() triggered")
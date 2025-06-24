class Memory:
    def __init__(self):
        return


class APICodeExecutor:
    def __init__(self, memory: Memory):
        self.memory = memory

    def init_memory(self):
        self.memory = Memory()

    # event parsing

    def run_calls(self, call_list: list[str]):
        for call in call_list:
            try:
                exec("self." + call)
            except Exception as e:
                print(f"[Error] Failed to run: {call}")
                print(f"        Reason: {e}")

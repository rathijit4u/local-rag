class MilvusHostError(Exception):
    def __init__(self, *args):
        super().__init__(f"Unable to find host - {args[0]}")
        
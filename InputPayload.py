class InputPayload:
    def __init__(self, data, data_type="text"):
        self.data = data
        self.data_type = data_type  # e.g., text, number, boolean, structured, image etc.

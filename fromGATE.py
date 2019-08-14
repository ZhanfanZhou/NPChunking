class GateExporter:
    def __init__(self, export_file):
        self. gate_pos = {}
        self.gate_token = {}

    def tokenize(self, id):
        return self.gate_token[id]

    def pos(self, id):
        return self.gate_pos[id]


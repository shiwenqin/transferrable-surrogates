from surrogates.encodings.base import BaseEncoder

class StrEncoder(BaseEncoder):
    def __init__(self):
        super().__init__()
    
    def encode_individual(self, individual):
        node = individual.root
        #return str(node)
        return node.to_long_string()

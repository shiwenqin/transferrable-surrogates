class BaseEncoder:
    def encode_individual(self, individual):
        raise NotImplementedError("encode_root method not implemented for the BaseEncoder")

    def encode(self, individual):        
        return self.encode_individual(individual)

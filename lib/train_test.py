class Trainer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        

class Tester:
    def __init__(self, trainer):
        self.trainer = trainer
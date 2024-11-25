import random


class Agent:
    def __init__(self, node_id, language):
        self.id = node_id
        self.language = language
        self.lr = random.random()

    def set_language(self, language):
        self.language = language

    def update_language(self, reference_language):
        new_language = dict()
        for key in reference_language:
            # Gradual adaptation: blend own and reference language
            if random.random() < self.lr:
                new_language[key] = reference_language[key]
            else:
                if key in self.language.concept_map:
                    new_language[key] = self.language.concept_map[key]
        self.language.concept_map = new_language
        self.lr *= 0.85

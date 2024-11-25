import random


class Language:
    def __init__(self, symbol_set, min_size, max_size):
        self.language_size = random.randint(min_size, max_size)
        self.concept_map = {}
        for i in range(self.language_size):
            word = random.choice(symbol_set)
            self.concept_map[word] = i

    def similarity(self, other_language):
        score = 0
        key_set = set(self.concept_map.keys())
        key_set.update(other_language.concept_map.keys())
        for word in self.concept_map:
            if word in other_language.concept_map and other_language.concept_map[word] == self.concept_map[word]:
                score += 1
        return score / len(key_set)

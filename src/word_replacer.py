import types
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
nltk.download('wordnet')
import pandas as pd
import random


class WordReplacer(object):
    
    def get_antonyms(self, word, pos=None):  
        antonyms = set()
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms.add(antonym.name())
        if word in antonyms:
            antonyms.remove(word)
        return list(antonyms)

    def get_synonyms(self,word):
        """
        Get synonyms of a word
        """
        synonyms = set()
    
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym)
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)

	
    def sentence_replacement(self,words,n,types=""):
        words = words.split()
        types= types.lower()
        new_words= words.copy()
        random_word_list = list(set([word for word in words if word not in stopwords.words("english")]))
        random.shuffle(random_word_list)
        num_replaced = 0
        if types == "antonyms":
            for random_word in random_word_list:
              antonyms = self.get_antonyms(random_word)

              if len(antonyms)>=1:
                antonyms = random.choice(list(antonyms))
                new_words = [antonyms if word == random_word else word for word in new_words]
                num_replaced +=1

              if num_replaced >=n:
                break
        
        if types=="synonyms":
            for random_word in random_word_list:
              synonyms = self.get_synonyms(random_word)

              if len(synonyms)>=1:
                synonyms = random.choice(list(synonyms))
                new_words = [synonyms if word == random_word else word for word in new_words]
                num_replaced +=1

              if num_replaced >=n:
                break
        sentence= " ".join(new_words)
        return sentence

class WordSwapping(object):
    
    def swap_word(self,new_words):
        random_idx_1 = random.randint(0, len(new_words)-1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words)-1)
            counter += 1

            if counter > 3:
                return new_words

        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
        return new_words

    def random_swap(self,words,n):
        words = words.split()
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        sentence = ' '.join(new_words)
        return sentence

# if __name__ == "__main__":
    # replace= WordReplacer()
    # temp1= ["i am testing", "this is second sent"]
    # print([replace.sentence_replacement(i,n=1,types="synonyms") for i in temp1])
     
    
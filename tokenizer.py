class PreambleTokenizer(object):
    # Init method
    def __init__(self, corpus):
        self.corpus = corpus
        self.encode_dict = self.dict_from_corpus(corpus)
        self.decode_dict = {v: k for k, v in self.encode_dict.items()}

    # Encoding dictionary

    def dict_from_corpus(self, corpus):
        # read in corpus from file
        with open(corpus, 'r') as file:
            text = file.read()
        # split corpus into words
        words = text.split()
        # remove duplicates
        words = list(dict.fromkeys(words))
        # sort words
        words.sort()
        # create dictionary
        return {word: i for i, word in enumerate(words)}


    """ENCODE_DICT = {
        'and': 0, 'to': 1, 'more': 2, 'insure': 3, 'do': 4, 'People': 5,
        'Tranquility,': 6, 'provide': 7, 'Order': 8, 'secure': 9, 'Liberty': 10,
        'domestic': 11, 'Posterity,': 12, 'common': 13, 'States,': 14, 'Welfare,': 15,
        'defence,': 16, 'the': 17, 'a': 18, 'for': 19, 'United': 20, 'Justice,': 21,
        'Constitution': 22, 'of': 23, 'in': 24, 'ordain': 25, 'ourselves': 26,
        'perfect': 27, 'form': 28, 'this': 29, 'general': 30, 'establish': 31,
        'Union,': 32, 'States': 33, 'We': 34, 'Blessings': 35, 'our': 36, 'America.': 37,
        'promote': 38
    }"""
    #ENCODE_DICT = dict_from_corpus("get_low.txt")


    # Decoding dictionary
    #DECODE_DICT = {v: k for k, v in ENCODE_DICT.items()}

    def encode(self, word):
        return self.encode_dict.get(word, None)

    def decode(self, token):
        return self.decode_dict.get(token, '')

    def encode_text(self, text):
        words = text.split()
        return [self.encode_dict.get(word, None) for word in words]

    def decode_tokens(self, tokens):
        return ' '.join([self.decode_dict.get(token, '') for token in tokens])


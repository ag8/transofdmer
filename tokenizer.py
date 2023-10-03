class PreambleTokenizer:
    # Encoding dictionary
    ENCODE_DICT = {
        'and': 0, 'to': 1, 'more': 2, 'insure': 3, 'do': 4, 'People': 5,
        'Tranquility,': 6, 'provide': 7, 'Order': 8, 'secure': 9, 'Liberty': 10,
        'domestic': 11, 'Posterity,': 12, 'common': 13, 'States,': 14, 'Welfare,': 15,
        'defence,': 16, 'the': 17, 'a': 18, 'for': 19, 'United': 20, 'Justice,': 21,
        'Constitution': 22, 'of': 23, 'in': 24, 'ordain': 25, 'ourselves': 26,
        'perfect': 27, 'form': 28, 'this': 29, 'general': 30, 'establish': 31,
        'Union,': 32, 'States': 33, 'We': 34, 'Blessings': 35, 'our': 36, 'America.': 37,
        'promote': 38
    }

    # Decoding dictionary
    DECODE_DICT = {v: k for k, v in ENCODE_DICT.items()}

    @classmethod
    def encode(cls, word):
        return cls.ENCODE_DICT.get(word, None)

    @classmethod
    def decode(cls, token):
        return cls.DECODE_DICT.get(token, None)

    @classmethod
    def encode_text(cls, text):
        words = text.split()
        return [cls.ENCODE_DICT.get(word, None) for word in words]

    @classmethod
    def decode_tokens(cls, tokens):
        return ' '.join([cls.DECODE_DICT.get(token, '') for token in tokens])


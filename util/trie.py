
from itertools import chain

import pygtrie


class TokenTrie:
    def __init__(self, tokenizer=None, end_of_seq=None):
        self.tokenizer = tokenizer
        self.end_of_seq = end_of_seq
        self.end_of_seq_id = tokenizer.convert_tokens_to_ids(end_of_seq)
        self.trie = pygtrie.Trie()

    def _normalize_id_seq(self, id_seq):
        if id_seq[-1] != self.end_of_seq_id:
            return tuple(chain(id_seq, [self.end_of_seq_id]))
        else:
            return tuple(id_seq)

    def add_id_seq(self, id_seq):
        self.trie[self._normalize_id_seq(id_seq)] = True

    def add_token_seq(self, token_seq):
        id_seq = self.tokenizer.convert_tokens_to_ids(token_seq)
        self.add_id_seq(id_seq)

    def add_text(self, text):
        tokenized = self.tokenizer(text, add_special_tokens=False)
        self.add_id_seq(tokenized['input_ids'])

    def __contains__(self, key):
        # "key" is a sequence of ids or tokens
        # return self.trie.get(key, False)
        if isinstance(key, (list, tuple)) and isinstance(key[0], str):
            id_seq = self.tokenizer.convert_tokens_to_ids(key)
        elif isinstance(key, str):
            tokenized = self.tokenizer(key, add_special_tokens=False)
            id_seq = tokenized['input_ids']
        else:
            assert isinstance(key[0], int)
            id_seq = key

        return self._normalize_id_seq(id_seq) in self.trie

    def candidate_ids(self, id_seq_prefix):
        # "id_seq_prefix" is a part of an entire sequence of a key
        prefix_node, path = self.trie._get_node(id_seq_prefix)
        return (token_id for token_id, node in prefix_node.children.iteritems())

    def candidate_tokens(self, token_seq_prefix):
        # "id_seq_prefix" is a part of an entire sequence of a key
        id_seq_prefix = self.tokenizer.convert_tokens_to_ids(token_seq_prefix)
        candidate_ids = self.candidate_ids(id_seq_prefix)
        return self.tokenizer.convert_ids_to_tokens(candidate_ids)

    def id_seqs(self):
        return iter(self.trie)

    def token_seqs(self):
        def ids_to_tokens(ids):
            return self.tokenizer.convert_ids_to_tokens(ids[:-1])

        return map(ids_to_tokens, self.id_seqs())

    __iter__ = token_seqs


if __name__ == '__main__':
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained(
        './pretrained/bart-base',
        add_prefix_space=True)
    end_of_seq = '<end-of-seq>'
    tokenizer.add_tokens([end_of_seq], special_tokens=True)
    trie = TokenTrie(tokenizer, end_of_seq)

    trie.add_text("I'm old.")
    trie.add_text("You have a cat.")

    print(tuple(trie))
    pass


from itertools import chain

from dhnamlib.pylib.iteration import all_same
from dhnamlib.pylib.decoration import construct

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

    def candidate_ids(self, id_seq_prefix, ignoring_errors=False):
        # "id_seq_prefix" is a part of an entire sequence of a key
        try:
            prefix_node, path = self.trie._get_node(id_seq_prefix)
        except KeyError as error:
            if ignoring_errors:
                candidate_ids = ()
            else:
                raise error
        else:
            candidate_ids = (token_id for token_id, node in prefix_node.children.iteritems())
        return candidate_ids

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

    @classmethod
    def merge(cls, token_trie):
        token_trie = tuple(token_trie)

        assert all_same(token_trie.tokenizer for token_trie in token_trie)
        assert all_same(token_trie.end_of_seq for token_trie in token_trie)

        merged_token_trie = cls(token_trie[0].tokenizer, token_trie[0].end_of_seq)

        for token_trie in token_trie:
            merged_token_trie.trie.update(token_trie.trie)

        return merged_token_trie

    __iter__ = token_seqs


class SpanTrie:
    def __init__(self, id_seq, end_of_seq_id):
        self.id_seq = id_seq    # id_seq does not include BOS and EOS. We also assume "add_prefix_space=True".
        self.end_of_seq_id = end_of_seq_id

        id_to_index_set = dict()
        for index, token_id in enumerate(id_seq):
            id_to_index_set.setdefault(token_id, set()).add(index)
        self.id_to_index_set = id_to_index_set

    @construct(lambda x: sorted(set(x)))
    def candidate_ids(self, id_seq_prefix):
        if len(id_seq_prefix) == 0:
            yield from self.id_seq
        else:
            token_id_iter = iter(id_seq_prefix)
            index_set = self.id_to_index_set.get(next(token_id_iter), set())
            for token_id in token_id_iter:
                if len(index_set) == 0:
                    break
                next_index_set = set()
                for index in index_set:
                    next_index = index + 1
                    if next_index < len(self.id_seq) and self.id_seq[next_index] == token_id:
                        next_index_set.add(next_index)
                index_set = next_index_set
            if len(index_set) > 0:
                yield self.end_of_seq_id
                for index in index_set:
                    next_index = index + 1
                    if next_index < len(self.id_seq):
                        yield self.id_seq[next_index]
            else:
                yield from []


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

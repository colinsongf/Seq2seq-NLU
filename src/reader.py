import numpy as np
import pickle
from collections import Counter
import os
from config import cfg

BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
PAD = "<pad>"
MAX_SENTENCE_LEN = 100


class Reader:
    def __init__(self, vocab_size, buckets):
        self.encoder_input = []
        self.decoder_input = []
        self.vocab_size = vocab_size
        self.vocab_dict = {}
        self.vocab_dict_rev = {}

        self.dataset_enc = []
        self.dataset_dec = []
        self.dec_weights = []

        self.dataset_enc_tok = []
        self.dataset_dec_tok = []

        self.buckets = buckets
        self.buckets_with_ids = [[] for _ in cfg["buckets"]]

        self.BOS_i = self.vocab_size - 4
        self.EOS_i = self.vocab_size - 3
        self.UNK_i = self.vocab_size - 2
        self.PAD_i = self.vocab_size - 1

    def build_dict(self, dictionary_path, reversed_dictionary_name, input_data_path):
        print("building dictionary...")

        if os.path.isfile(dictionary_path) and os.path.isfile(reversed_dictionary_name):
            self.vocab_dict = pickle.load(open(dictionary_path, "rb"))
            self.vocab_dict_rev = pickle.load(open(reversed_dictionary_name, "rb"))
            return

        cnt = Counter()
        with open(input_data_path, 'r') as input_data_file:
            for sentence in input_data_file:
                for word in sentence.split():
                    if word not in {BOS, EOS, UNK, PAD}:
                        cnt[word] += 1

            vocab_with_counts = cnt.most_common(self.vocab_size - 4)
            vocab = [i[0] for i in vocab_with_counts]
            ids = list(range(self.vocab_size - 4))
            self.vocab_dict = dict(list(zip(vocab, ids)))
            self.vocab_dict_rev = dict(list(zip(ids, vocab)))

            self.vocab_dict[BOS] = self.BOS_i
            self.vocab_dict[EOS] = self.EOS_i
            self.vocab_dict[UNK] = self.UNK_i
            self.vocab_dict[PAD] = self.PAD_i

            self.vocab_dict_rev[self.BOS_i] = BOS
            self.vocab_dict_rev[self.EOS_i] = EOS
            self.vocab_dict_rev[self.UNK_i] = UNK
            self.vocab_dict_rev[self.PAD_i] = PAD

            pickle.dump(self.vocab_dict, open(dictionary_path, "wb"))
            pickle.dump(self.vocab_dict_rev, open(reversed_dictionary_name, "wb"))

    def read_data(self, path):
        """
        Return value, assuming the path contains N sentences:
        Two lists of length 2N (dataset_enc and dataset_dec)
        The second sentence of every pair begins with bos
        """

        print("reading data from %s..." % path)
        with open(path, 'r') as f:

            for data_line in f.readlines():
                input_sentences = data_line.split('\t')
                assert (len(input_sentences) == 3)

                first_sentence_as_ids = self.ids_from_toks(input_sentences[0].split())
                second_sentence_as_ids = self.ids_from_toks(input_sentences[1].split())
                third_sentence_as_ids = self.ids_from_toks(input_sentences[2].split())

                first_sentence_as_ids = first_sentence_as_ids + [self.EOS_i]
                second_sentence_as_ids1 = [self.BOS_i] + second_sentence_as_ids + [self.EOS_i]
                second_sentence_as_ids2 = second_sentence_as_ids + [self.EOS_i]
                third_sentence_as_ids = [self.BOS_i] + third_sentence_as_ids + [self.EOS_i]

                first_sentence_as_toks = input_sentences[0].split()
                second_sentence_as_toks = input_sentences[1].split()
                third_sentence_as_toks = input_sentences[2].split()

                first_sentence_as_toks = first_sentence_as_toks + [EOS]
                second_sentence_as_toks1 = [BOS] + second_sentence_as_toks + [EOS]
                second_sentence_as_toks2 = second_sentence_as_toks + [EOS]
                third_sentence_as_toks = [BOS] + third_sentence_as_toks + [EOS]

                if len(first_sentence_as_ids) <= MAX_SENTENCE_LEN and len(second_sentence_as_ids1) <= MAX_SENTENCE_LEN:
                    self.dataset_enc.append(first_sentence_as_ids)
                    self.dataset_dec.append(second_sentence_as_ids1)
                    self.dec_weights.append(np.ones(shape=len(second_sentence_as_ids1), dtype=np.float32))

                    self.dataset_enc_tok.append(first_sentence_as_toks)
                    self.dataset_dec_tok.append(second_sentence_as_toks1)

                if len(second_sentence_as_ids2) <= MAX_SENTENCE_LEN and len(third_sentence_as_ids) <= MAX_SENTENCE_LEN:
                    self.dataset_enc.append(second_sentence_as_ids2)
                    self.dataset_dec.append(third_sentence_as_ids)
                    self.dec_weights.append(np.ones(shape=len(third_sentence_as_ids), dtype=np.float32))

                    self.dataset_enc_tok.append(second_sentence_as_toks2)
                    self.dataset_dec_tok.append(third_sentence_as_toks)

            for i in range(0, len(self.dataset_enc)):
                for bucket_index, bucket_size in enumerate(self.buckets):
                    if len(self.dataset_enc[i]) <= bucket_size and len(self.dataset_dec[i]) <= bucket_size:
                        self.dataset_enc[i].extend((bucket_size - len(self.dataset_enc[i])) * [self.PAD_i])
                        self.dataset_dec[i].extend((bucket_size - len(self.dataset_dec[i])) * [self.PAD_i])

                        self.buckets_with_ids[bucket_index].append(i)

                        self.dataset_enc_tok[i].extend((bucket_size - len(self.dataset_enc_tok[i])) * [PAD])
                        self.dataset_dec_tok[i].extend((bucket_size - len(self.dataset_dec_tok[i])) * [PAD])

                        break

            f.close()

    def ids_from_toks(self, tokens):
        ids = []
        for t in tokens:
            if self.vocab_dict.get(t) is not None:
                ids.append(self.vocab_dict.get(t))
            else:
                ids.append(self.UNK_i)
        return ids

    def toks_from_ids(self, ids):
        toks = []
        for i in ids:
            toks.append(self.vocab_dict_rev.get(i))
        return toks
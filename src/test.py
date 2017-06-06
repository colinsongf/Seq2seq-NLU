import pickle
from reader import Reader
from config import cfg

vocab_dict = pickle.load(open("../dicts/dictionary.p", "rb"))
vocab_dict_rev = pickle.load(open("../dicts/dictionary_rev.p", "rb"))


def output_dictionaries():
    with open("../output/vocab_dict", "w") as f:
        for key in vocab_dict:
            f.write(key + " " + str(vocab_dict[key]) + "\n")

        f.close()

    with open("../output/vocab_dict_rev", "w") as f:
        for key in vocab_dict_rev:
            f.write(vocab_dict_rev[key] + " " + str(key) + "\n")

        f.close()


def output_encoder_and_decoder():
    reader = Reader(cfg['vocab_size'], cfg['buckets'])
    reader.build_dict(cfg['dictionary_name'], cfg['reversed_dictionary_name'], cfg['path']['train'])
    reader.read_data(cfg['path']['train'])

    encoder_inputs = reader.dataset_enc
    with open("../output/encoder_inputs", "w") as f:
        for line in encoder_inputs:
            for word in line:
                f.write(str(word) + " ")
            f.write("\n")
        f.close()

    decoder_inputs = reader.dataset_dec
    with open("../output/decoder_inputs", "w") as f:
        for line in decoder_inputs:
            for word in line:
                f.write(str(word) + " ")
            f.write("\n")
        f.close()


output_encoder_and_decoder()

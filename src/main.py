import reader as our_reader
import cornell_data as cornell_reader
from config import cfg
import model


def main():
    reader1 = our_reader.Reader(cfg['vocab_size'], cfg['buckets'])
    reader2 = cornell_reader.Reader(cfg)

    reader1.build_dict(cfg['dictionary_name'], cfg['reversed_dictionary_name'], cfg['path']['train'])
    reader1.read_data(cfg['path']['train'])

    encoder_inputs_10 = [i for i in reader.dataset_enc if len(i) == 10]
    decoder_inputs_10 = [i for i in reader.dataset_dec if len(i) == 10]
    encoder_inputs__toks_10 = [i for i in reader.dataset_enc_tok if len(i) == 10]
    decoder_inputs__toks_10 = [i for i in reader.dataset_dec_tok if len(i) == 10]

    print("done")

    model.create_placeholders()



if __name__ == "__main__":
    main()

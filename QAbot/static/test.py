from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import glob
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils_seq2seq import Seq2SeqDecodeDataset
from cytoolz import curry
from tokenization_unilm import UnilmTokenizer, WhitespaceTokenizer
from modeling_unilm import UnilmForSeq2SeqDecode, UnilmConfig
import utils_seq2seq

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in (UnilmConfig,)), ())
MODEL_CLASSES = {
    'unilm': (UnilmConfig, UnilmForSeq2SeqDecode, UnilmTokenizer)
}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


@curry
def decode_coll(batch, bi_uni_pipeline, max_src_length):
    # truncate batch
    original_max_len = max([len(src_tokens) for json_id, src_tokens in batch])
    batch = [(json_id, src_tokens[:max_src_length]) for json_id, src_tokens in batch]
    batch_max_src_length = max([len(src_tokens) for json_id, src_tokens in batch])
    # sort batch by length
    batch = sorted(batch, key=lambda x: -len(x[1]))
    processed_batch = []
    json_id_list = []
    for json_id, src_tokens in batch:
        for proc in bi_uni_pipeline:
            processed_batch.append(proc((src_tokens, batch_max_src_length)))
        json_id_list.append(json_id)
    assert len(json_id_list) == len(processed_batch)
    processed_batch_tensors = utils_seq2seq.batch_list_to_batch_tensors(
        processed_batch)
    return processed_batch_tensors, json_id_list


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def answer(sentence):
    print("start")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(123)

    config_class, model_class, tokenizer_class = MODEL_CLASSES['unilm']
    config = config_class.from_pretrained(
        "E:/py_prog/UMbot/QAbot/", max_position_embeddings=512)
    tokenizer = tokenizer_class.from_pretrained(
        "E:/py_prog/UMbot/QAbot/", do_lower_case=False)

    bi_uni_pipeline = []
    bi_uni_pipeline.append(utils_seq2seq.Preprocess4Seq2seqDecode(list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids,
                                                                  512, max_tgt_length=256))

    # Prepare data loader
    max_src_length =254
    decode_dataset = Seq2SeqDecodeDataset(sentence, 1, tokenizer, 512)
    decode_dataset_sampler = SequentialSampler(decode_dataset)
    decode_dataloader = DataLoader(decode_dataset, sampler=decode_dataset_sampler, batch_size=1,
                                    collate_fn=decode_coll(bi_uni_pipeline=bi_uni_pipeline, max_src_length=max_src_length))

    # Prepare model
    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        ["[MASK]", "[SEP]", "[S2S_SOS]"])
    forbid_ignore_set = None
    model_recover_path_list = glob.glob("E:/py_prog/UMbot/QAbot/model.best.bin")
    model_recover_path_list.sort()
    for model_recover_path in model_recover_path_list:
        model_recover = torch.load(model_recover_path)
        model = model_class.from_pretrained("E:/py_prog/UMbot/QAbot/", state_dict=model_recover, config=config, mask_word_id=mask_word_id, search_beam_size=5, length_penalty=0,
                                            eos_id=eos_word_ids, sos_id=sos_word_id, forbid_duplicate_ngrams=False, forbid_ignore_set=forbid_ignore_set, ngram_size=3, min_len=None)
        del model_recover

        model.to(device)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        torch.cuda.empty_cache()
        model.eval()
        output_list = []
        j=0
        for batch_i, batch in enumerate(tqdm(decode_dataloader)):
            with torch.no_grad():
                model_input, json_id_list = batch
                batch_size = len(json_id_list)
                model_input = [
                    t.to(device) if t is not None else None for t in model_input]
                input_ids, token_type_ids, position_ids, input_mask = model_input
                traces = model(input_ids, token_type_ids,
                               position_ids, input_mask)
                traces = {k: v.tolist() for k, v in traces.items()}
                output_ids = traces['pred_seq']

                for i in range(batch_size):
                    w_ids = output_ids[i]
                    output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                    output_tokens = []
                    for t in output_buf:
                        if t in ("[SEP]", "[PAD]","。"):
                            break
                        output_tokens.append(t)
                    output_sequence = ''.join(detokenize(output_tokens))
                    print(output_sequence)
                    output_list.append(output_sequence)
                    j=j+1
        return output_list[0]

def main():
    answer('提车验车时要注意哪些事项')

if __name__ == "__main__":
    main()
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import glob
import argparse
import math
import random
from tqdm import tqdm, trange
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import json
from tokenization_unilm import UnilmTokenizer, WhitespaceTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from modeling_unilm import UnilmForSeq2SeqDecode, UnilmConfig
# from transformers import (UnilmTokenizer, WhitespaceTokenizer,
#                           UnilmForSeq2SeqDecode, AdamW, UnilmConfig)


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


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def answer(sentence):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(123)

    config_class, model_class, tokenizer_class = MODEL_CLASSES["unilm"]
    config = config_class.from_pretrained(
        os.path.join(os.getcwd(),"QAbot"), max_position_embeddings=512)
    tokenizer = tokenizer_class.from_pretrained(
        os.path.join(os.getcwd(),"QAbot"), do_lower_case=False)

    bi_uni_pipeline = []
    bi_uni_pipeline.append(utils_seq2seq.Preprocess4Seq2seqDecode(list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids,
                                                                  512, max_tgt_length=256))

    # Prepare model
    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        ["[MASK]", "[SEP]", "[S2S_SOS]"])
    forbid_ignore_set = None
    for model_recover_path in glob.glob(os.path.join(os.getcwd(),"QAbot\\model.12.bin")):
        logger.info("***** Recover model: %s *****", model_recover_path)
        model_recover = torch.load(model_recover_path)
        model = model_class.from_pretrained(os.path.join(os.getcwd(),"QAbot"), state_dict=model_recover, config=config, mask_word_id=mask_word_id, search_beam_size=5, length_penalty=0,
                                            eos_id=eos_word_ids, sos_id=sos_word_id, forbid_duplicate_ngrams=False, forbid_ignore_set=forbid_ignore_set, ngram_size=3, min_len=None)
        del model_recover

        model.to(device)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        torch.cuda.empty_cache()
        model.eval()
        next_i = 0
        max_src_length = 254
        input_lines = [sentence]
        data_tokenizer = tokenizer
        input_lines = [['[CLS]'] + data_tokenizer.tokenize(
            x)[:max_src_length] + ['[SEP]'] for x in input_lines]
        #input_lines = sorted(list(enumerate(input_lines)),
        #                     key=lambda x: -len(x[1]))
        input_lines = list(enumerate(input_lines))
        output_lines = []
        score_trace_list = [None] * len(input_lines)
        total_batch = math.ceil(len(input_lines) / 1)
        
        with tqdm(total=total_batch) as pbar:
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + 1]
                buf_id = [x[0] for x in _chunk]
                buf = [x[1] for x in _chunk]
                next_i += 1
                max_a_len = max([len(x) for x in buf])
                instances = []
                for instance in [(x, max_a_len) for x in buf]:
                    for proc in bi_uni_pipeline:
                        instances.append(proc(instance))
                with torch.no_grad():
                    batch = utils_seq2seq.batch_list_to_batch_tensors(
                        instances)
                    batch = [
                        t.to(device) if t is not None else None for t in batch]
                    input_ids, token_type_ids, position_ids, input_mask = batch
                    traces = model(input_ids, token_type_ids,
                                   position_ids, input_mask)
                    traces = {k: v.tolist() for k, v in traces.items()}
                    output_ids = traces['pred_seq']
                    for i in range(len(buf)):
                        w_ids = output_ids[i]
                        output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                        output_tokens = []
                        for t in output_buf:
                            if t in ("[SEP]", "[PAD]","。"):
                                break
                            output_tokens.append(t)
                        output_sequence = ''.join(detokenize(output_tokens))
                        print(output_sequence)
                        output_lines.append(output_sequence)
                pbar.update(1)
    return output_lines[0]


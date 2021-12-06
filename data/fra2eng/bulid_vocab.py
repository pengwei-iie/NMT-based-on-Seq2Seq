# -*- coding: utf-8 -*-
from collections import Counter
import argparse

# Extra vocabulary symbols
# pad_token = "<pad>"
# unk_token = "<unk>"
# bos_token = "<bos>"
# eos_token = "<eos>"
# sep_token = "<sep>"
#
# extra_tokens = [pad_token, unk_token, bos_token, eos_token, sep_token]
#
# PAD = extra_tokens.index(pad_token)
# UNK = extra_tokens.index(unk_token)
# BOS = extra_tokens.index(bos_token)
# EOS = extra_tokens.index(eos_token)
# SEP = extra_tokens.index(sep_token)


def read_parallel_corpus(src_path, max_len, lower_case=False):
    print ('Reading examples from {}..'.format(src_path))
    src_sents, tgt_sents = [], []
    empty_lines, exceed_lines = 0, 0
    with open(src_path, encoding='utf-8') as src_file:
        for idx, src_line in enumerate(src_file):
            src_line, tgt_line = src_line.strip().split('\t')
            if idx % 1000 == 0:
                print('  reading {} lines..'.format(idx))
            if src_line.strip() == '':  # remove empty lines
                empty_lines += 1
                continue
            if lower_case:  # check lower_case
                src_line = src_line.lower()
                tgt_line = tgt_line.lower()
            # deal with Eng
            src_words = src_line.strip().split()
            tgt_words = tgt_line.strip().split()

            # deal with Chinese
            # src_words = list(src_line.strip().split()[0])
            # tgt_words = list(tgt_line.strip().split()[0])
            # replace & -> [sep]
            # src_words = ['[sep]' if i == '&' else i for i in src_words]

            if max_len is not None and (len(src_words) > max_len or len(tgt_words) > max_len):
                exceed_lines += 1
                continue
            src_sents.append(src_words)
            tgt_sents.append(tgt_words)

    print ('Filtered {} empty lines'.format(empty_lines),
           'and {} lines exceeding the length {}'.format(exceed_lines, max_len))
    print ('Result: {} lines remained'.format(len(src_sents)))
    return src_sents, tgt_sents


def build_vocab(examples, max_size, min_freq, save_vocab):
    print ('Creating vocabulary with max limit {}..'.format(max_size))
    counter = Counter()
    word2idx, idx2word = {}, []

    min_freq = max(min_freq, 1)
    max_size = max_size + len(idx2word) if max_size else None
    for sent in examples:
        for w in sent:
            counter.update([w])
    # first sort items in alphabetical order and then by frequency
    sorted_counter = sorted(counter.items(), key=lambda tup: tup[0])
    sorted_counter.sort(key=lambda tup: tup[1], reverse=True)
    with open(save_vocab, 'w', encoding='utf-8') as out_file:
        for word, freq in sorted_counter:
            if freq < min_freq or (max_size and len(idx2word) == max_size):
                break
            out_file.write(word)
            out_file.write('\n')
            idx2word.append(word)
            word2idx[word] = len(idx2word) - 1

    print('Vocabulary of size {} has been created'.format(len(idx2word)))
    return counter, word2idx, idx2word

def main(opt):
    train_src, train_tgt = read_parallel_corpus(opt.train_src, opt.max_len, opt.lower_case)
    counter, word2idx, idx2word = build_vocab(train_src, opt.src_vocab_size,
                                              opt.min_word_count, opt.save_src_vocab)
    counter, word2idx, idx2word = build_vocab(train_tgt, opt.tgt_vocab_size,
                                              opt.min_word_count_tgt, opt.save_tgt_vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing')

    parser.add_argument('-train_src', required=True, type=str, help='Path to training source data')
    parser.add_argument('-vocab', type=str, help='Path to an existing vocabulary file')
    parser.add_argument('-src_vocab_size', type=int, default=5000, help='Source vocabulary size')
    parser.add_argument('-tgt_vocab_size', type=int, default=5000, help='Target vocabulary size')
    parser.add_argument('-min_word_count', type=int, default=1)
    parser.add_argument('-min_word_count_tgt', type=int, default=1)
    parser.add_argument('-max_len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('-lower_case', action='store_true')
    parser.add_argument('-save_src_vocab', required=True, type=str, help='Output file for the src vocab')
    parser.add_argument('-save_tgt_vocab', required=True, type=str, help='Output file for the tgt vocab')

    opt = parser.parse_args()
    print(opt)
    main(opt)


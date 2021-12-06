DEVICES="cpu"

TRAIN_PATH=data/fra2eng/fra_eng.pairs
DEV_PATH=data/fra2eng/fra_eng.dev
SRC_VOCAB=data/fra2eng/src_vocab_file
TGT_VOCAB=data/fra2eng/tgt_vocab_file
checkpoint="experiment/4000.pt"

# Start inference
python runModel.py \
        --device $DEVICES \
        --train_path $TRAIN_PATH \
        --dev_path $DEV_PATH \
        --src_vocab_file $SRC_VOCAB \
        --tgt_vocab_file $TGT_VOCAB \
        --bidirectional \
        --use_attn \
        --load_checkpoint $checkpoint \
        --phase infer \
        --beam_width 5  \
         --batch_size 1

# python runModel.py --device cpu --train_path data/fra2eng/fra_eng.pairs --dev_path data/fra2eng/fra_eng.dev --src_vocab_file data/fra2eng/src_vocab_file --tgt_vocab_file data/fra2eng/tgt_vocab_file --bidirectional --use_attn --load_checkpoint "/home/pengwei.pw/third_next/classtanxinnmt/seq2seq/experiment/best/331.pt" --phase infer --beam_width 1

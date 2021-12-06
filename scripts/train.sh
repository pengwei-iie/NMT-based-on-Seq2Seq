DEVICES="0"

TRAIN_PATH=data_commongen/fra2eng/fra_eng.train
DEV_PATH=data_commongen/fra2eng/fra_eng.valid
SRC_VOCAB=data_commongen/fra2eng/src_vocab_file
TGT_VOCAB=data_commongen/fra2eng/tgt_vocab_file

# Start training
python runModel.py \
        --device $DEVICES \
        --train_path $TRAIN_PATH \
        --dev_path $DEV_PATH \
        --src_vocab_file $SRC_VOCAB \
        --tgt_vocab_file $TGT_VOCAB \
        --bidirectional \
        --use_attn \
        --random_seed 2808 \
        # --resume

# python runModel.py --device cpu --train_path data/fra2eng/fra_eng.pairs --dev_path data/fra2eng/fra_eng.dev --src_vocab_file data/fra2eng/src_vocab_file --tgt_vocab_file data/fra2eng/tgt_vocab_file --bidirectional --use_attn --random_seed 2808

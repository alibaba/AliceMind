#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

ROOT=$(dirname "$0")
echo "ROOT: $ROOT"
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

src=en
tgt=de
lang=en-de


URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz"
    "http://data.statmt.org/wmt17/translation-task/dev.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v12.tgz"
    "dev.tgz"
    "test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "training/news-commentary-v12.de-en"
)

# This will make the dataset compatible to the one used in "Convolutional Sequence to Sequence Learning"
# https://arxiv.org/abs/1705.03122
if [ "$1" == "--icml17" ]; then
    URLS[2]="http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
    FILES[2]="training-parallel-nc-v9.tgz"
    CORPORA[2]="training/news-commentary-v9.de-en"
    OUTDIR=wmt14.en_de
else
    OUTDIR=wmt17.en_de
fi

orig="$ROOT/data/wmt/orig"
OUTPUT_DIR="$ROOT/data/$OUTDIR"
tmp=$OUTPUT_DIR/tmp
raw=$OUTPUT_DIR/raw
prep=$OUTPUT_DIR/bpe
bin=$OUTPUT_DIR/bin

mkdir -p $orig $tmp $raw $prep $bin

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

cd $orig
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        fi
    fi
done
cd $ROOT

echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR >> $tmp/train.tags.$lang.tok.$l
#            perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
    done
done

## take newstest2013 as dev
#echo "pre-processing dev data..."
#for l in $src $tgt; do
#    if [ "$l" == "$src" ]; then
#        t="src"
#    else
#        t="ref"
#    fi
#    # newstest2013-ref.cs.sgm
#    grep '<seg id' $orig/dev/newstest2013-$t.$l.sgm | \
#        sed -e 's/<seg id="[0-9]*">\s*//g' | \
#        sed -e 's/\s*<\/seg>\s*//g' | \
#        sed -e "s/\’/\'/g" > $raw/valid_newstest2013.$l
##    perl $TOKENIZER -threads 8 -a -l $l > $tmp/valid.$l
#    echo ""
#done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test-full/newstest2014-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > $raw/test.$l
#    perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
    echo ""
done

echo "splitting train and valid..."
for l in $src $tgt; do
    awk '{if (NR%100 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $raw/valid.$l
    awk '{if (NR%100 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $raw/train.$l
done


SCRIPTS=$ROOT/../../scripts
SPM_ENCODE=$SCRIPTS/spm_encode.py
SPM_MODEL_PATH=$ROOT/../../saved_model/veco-large/sentencepiece.bpe.model
TRAIN_MINLEN=1  # remove sentences with <1 BPE token
TRAIN_MAXLEN=250  # remove sentences with >250 BPE tokens

echo "encoding train with learned BPE for train and valid ..."
for mode in 'train' 'valid'; do
    echo "apply_bpe.py to ${f}..."
    python "$SPM_ENCODE" \
        --model "$SPM_MODEL_PATH" \
        --output_format=piece \
        --inputs  $raw/$mode.$src $raw/$mode.$tgt \
        --outputs $prep/bpe.$mode.$src $prep/bpe.$mode.$tgt \
        --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN
done

#for mode in 'valid_newstest2013'; do
#    echo "apply_bpe.py to ${f}..."
#    python "$SPM_ENCODE" \
#        --model "$SPM_MODEL_PATH" \
#        --output_format=piece \
#        --inputs  $raw/$mode.$src $raw/$mode.$tgt \
#        --outputs $prep/bpe.$mode.$src $prep/bpe.$mode.$tgt \
#        --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN
#done

echo "encoding train with learned BPE for test ..."
for mode in 'test'; do
    echo "apply_bpe.py to ${f}..."
    python "$SPM_ENCODE" \
        --model "$SPM_MODEL_PATH" \
        --output_format=piece \
        --inputs  $raw/$mode.$src $raw/$mode.$tgt \
        --outputs $prep/bpe.$mode.$src $prep/bpe.$mode.$tgt
done


wc -l $prep/bpe.train*
wc -l $prep/bpe.valid*
wc -l $prep/bpe.test*

cd ../../
pwd

DICT=$OUTPUT_DIR/dict/dict.txt  # copy from saved_model/veco-large/dict.txt
echo " ==== Convert bpe tokens to ids > ${bin} ..."
python preprocess.py \
    --source-lang ${src} \
    --target-lang ${tgt} \
    --trainpref ${prep}/bpe.train \
    --validpref ${prep}/bpe.valid \
    --testpref ${prep}/bpe.test  \
    --destdir ${bin} \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --srcdict ${DICT} \
    --tgtdict ${DICT} \
    --workers 70

## Use resized dictionary to binarize data
#python preprocess.py \
#    --source-lang en \
#    --target-lang de \
#    --trainpref ${prep}/bpe.train \
#    --validpref ${prep}/bpe.valid \
#    --testpref ${prep}/bpe.test  \
#    --destdir ${bin}-resized_dict \
#    --thresholdtgt 0 \
#    --thresholdsrc 0 \
#    --srcdict $OUTPUT_DIR/dict/resized_dict.txt \
#    --tgtdict $OUTPUT_DIR/dict/resized_dict.txt \
#    --workers 70
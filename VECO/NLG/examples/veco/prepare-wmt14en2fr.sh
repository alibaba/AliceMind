#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

ROOT=$(dirname "$0")
echo "ROOT: $ROOT"
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://statmt.org/wmt13/training-parallel-un.tgz"
    "http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
    "http://statmt.org/wmt10/training-giga-fren.tar"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-un.tgz"
    "training-parallel-nc-v9.tgz"
    "training-giga-fren.tar"
    "test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.fr-en"
    "commoncrawl.fr-en"
    "un/undoc.2000.fr-en"
    "training/news-commentary-v9.fr-en"
    "giga-fren.release2.fixed"
)

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=fr
lang=en-fr
orig="$ROOT/data/wmt/orig"
OUTPUT_DIR="$ROOT/data/wmt14.en_fr"
tmp=$OUTPUT_DIR/tmp
raw=$OUTPUT_DIR/raw
prep=$OUTPUT_DIR/bpe
bin=$OUTPUT_DIR/bin

mkdir -p $orig $tmp $raw $prep $bin

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

gunzip giga-fren.release2.fixed.*.gz
cd $ROOT

echo "pre-processing train data..."
for l in $src $tgt; do
    rm $raw/train.$l
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR >> $raw/train.$l
    done
done


echo "pre-processing valid data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    # valid set = newstest2012+newstest2013
    grep '<seg id' $orig/dev/newstest2012-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > $raw/valid.$l
    grep '<seg id' $orig/dev/newstest2013-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" >> $raw/valid.$l

    echo ""
done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    # test set = newstest2014
    grep '<seg id' $orig/test-full/newstest2014-fren-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > $raw/test.$l
#    perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
    echo ""
done

wc $raw/* -l

SCRIPTS=$ROOT/../../scripts
SPM_ENCODE=$SCRIPTS/spm_encode.py
SPM_MODEL_PATH=$ROOT/../../saved_model/veco-large/sentencepiece.bpe.model
TRAIN_MINLEN=1  # remove sentences with <1 BPE token
TRAIN_MAXLEN=250  # remove sentences with >250 BPE tokens


echo "encoding train with learned BPE for train and valid ..."
for mode in 'train' 'valid' 'test'; do
    echo "apply_bpe.py to $raw/$mode.$src $raw/$mode.$tgt ..."
    python "$SPM_ENCODE" \
        --model "$SPM_MODEL_PATH" \
        --output_format=piece \
        --inputs  $raw/$mode.$src $raw/$mode.$tgt \
        --outputs $raw/bpe.$mode.$src $raw/bpe.$mode.$tgt
done

perl $CLEAN -ratio 1.5 $raw/bpe.train $src $tgt $prep/bpe.train 1 250
perl $CLEAN -ratio 1.5 $raw/bpe.valid $src $tgt $prep/bpe.valid 1 250

for L in $src $tgt; do
    cp $raw/bpe.test.$L $prep/bpe.test.$L
done


wc -l $prep/bpe.train*
wc -l $prep/bpe.valid*
wc -l $prep/bpe.test*

cd ../../
pwd

# Use the full dictionary to binarize data
DICT=$OUTPUT_DIR/dict/dict.txt   # copy from saved_model/veco-large/dict.txt
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

## Use the resized dictionary to binarize data (Resized  dict size: 100816)
#python preprocess.py \
#    --source-lang en \
#    --target-lang fr \
#    --trainpref ${prep}/bpe.train \
#    --validpref ${prep}/bpe.valid \
#    --testpref ${prep}/bpe.test  \
#    --destdir ${bin}-resized_dict \
#    --thresholdtgt 0 \
#    --thresholdsrc 0 \
#    --srcdict $OUTPUT_DIR/dict/resized_dict.txt \
#    --tgtdict $OUTPUT_DIR/dict/resized_dict.txt \
#    --workers 70

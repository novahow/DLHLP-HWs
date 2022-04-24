#!/bin/bash

. ./path.sh || exit 1;

stage=$1
stop_stage=$2
model_name=ctc_libri-10h_char
config=$3
ckpt= #model/ctc_libri-10h_char/epoch=234-step=40654.ckpt # model/ctc_libri-10h_char/epoch=44-step=3869.ckpt


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Download data"
    mkdir -p data
    cd data
    wget https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz
    # wget https://www.openslr.org/resources/12/dev-clean.tar.gz --no-check-certificate
    # wget https://www.openslr.org/resources/12/test-clean.tar.gz --no-check-certificate

    tar zxf librispeech_finetuning.tgz
    # tar zxf dev-clean.tar.gz
    # tar zxf test-clean.tar.gz

    rm librispeech_finetuning.tgz
    # rm dev-clean.tar.gz
    # rm test-clean.tar.gz

    cd ..
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Preprocess data"

    run_preprocess.py \
        -c LibriSpeech \
        -p data/librispeech_finetuning \
        -s 1h \
        -o data/libri_train_1h \
        --gen-vocab \
        --char-vocab-size 40

    run_preprocess.py \
        -c LibriSpeech \
        -p data/librispeech_finetuning \
        -s 9h \
        -o data/libri_train_9h

    run_preprocess.py \
        -c LibriSpeech \
        -p data/LibriSpeech \
        -s dev-clean \
        -o data/libri_dev

    run_preprocess.py \
        -c LibriSpeech \
        -p data/LibriSpeech \
        -s test-clean \
        -o data/libri_test
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Train E2E ASR"
    mkdir -p model

    if [ -z "$ckpt" ]; then
        run_asr.py \
            --config ${config} \
            --override "args.trainer.default_root_dir=\"model/${model_name}\""
    else
        echo "Resume training from $ckpt"
        run_asr.py \
            --ckpt $ckpt --config ${config}
    fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Test E2E ASR"
    mkdir -p model

    if [ -z "$ckpt" ]; then
        echo "Error: ckpt must be specified during testing!" || exit 1
    else
        run_asr.py \
            --config config/test.yaml \
            --test \
            --ckpt $ckpt
    fi
fi

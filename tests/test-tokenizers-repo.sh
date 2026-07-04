#!/usr/bin/env bash

if [ $# -lt 2 ]; then
    printf "Usage: $0 <git-repo> <target-folder> [<test-exe>] [<revision>]\n"
    exit 1
fi

if [ $# -ge 3 ]; then
    toktest=$3
else
    toktest="./test-tokenizer-0"
fi

# optional revision to pin the vocab repo to (newer vocabs may need newer llama.cpp)
revision=${4:-}

if [ ! -x $toktest ]; then
    printf "Test executable \"$toktest\" not found!\n"
    exit 1
fi

repo=$1
folder=$2

if [ -d $folder ] && [ -d $folder/.git ]; then
    if [ -n "$revision" ]; then
        (cd $folder; git fetch --quiet origin; git checkout --quiet --force $revision)
    else
        (cd $folder; git pull)
    fi
else
    git clone $repo $folder
    if [ -n "$revision" ]; then
        (cd $folder; git checkout --quiet --force $revision)
    fi

    # byteswap models if on big endian
    if [ "$(uname -m)" = s390x ]; then
        for f in $folder/*/*.gguf; do
            echo YES | python3 "$(dirname $0)/../gguf-py/gguf/scripts/gguf_convert_endian.py" $f big
        done
    fi
fi

shopt -s globstar
for gguf in $folder/**/*.gguf; do
    if [ -f $gguf.inp ] && [ -f $gguf.out ]; then
        $toktest $gguf
    else
        printf "Found \"$gguf\" without matching inp/out files, ignoring...\n"
    fi
done


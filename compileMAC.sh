#!/bin/bash

# make sure pybind11 is installed : `pip3 install pybind11`

# `python3-config --ldflags` try to limit stack_size and clang
# does not like that when compiling a shared library, hence
# we remove it.
LD_FLAGS=`python3-config --ldflags | sed 's/-Wl,-stack_size,[0-9]*//g'`
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` golois.cpp \
    -o golois`python3-config --extension-suffix` \
    `python3-config --includes` $LD_FLAGS \
    `python3-config --cflags`

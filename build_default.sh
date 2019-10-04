#!/usr/bin/env bash
cd caffe
# fix issue with caffe build from source in ubuntu, see more details here: https://github.com/BVLC/caffe/issues/2347
find . -type f -exec sed -i -e 's^"hdf5.h"^"hdf5/serial/hdf5.h"^g' -e 's^"hdf5_hl.h"^"hdf5/serial/hdf5_hl.h"^g' '{}' \;

# if Makefile.config has already existed, then don't overwrite it
if [ ! -e  Makefile.config ]; then
    cp Makefile.config.example Makefile.config
fi
make clean
make all pycaffe
cd ..
cp settings_user.py.example settings_user.py

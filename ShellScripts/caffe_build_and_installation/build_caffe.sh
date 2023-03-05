#!/bin/bash
#git clone https://github.com/BVLC/caffe.git

#!/bin/bash
#git clone https://github.com/BVLC/caffe.git

#sudo apt update -y
#sudo apt install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler \
#        libboost-all-dev libatlas-base-dev libopenblas-dev libgflags-dev libgoogle-glog-dev liblmdb-dev

source env3.6/bin/activate
cd python
#echo "export PATH=\$PATH:$(pwd)" >> ~/.bashrc
pip install -r requirements.txt
cd ..

make clean
export CPLUS_INCLUDE_PATH=/usr/include/python3.6m/
export PYTHON_LIBRARIES=/usr/lib/x86_64-linux-gnu/

cat >> Makefile.config << EOF
CPU_ONLY := 1
OPENCV_VERSION := 3
CUSTOM_CXX := g++
WITH_PYTHON_LAYER := 1
BLAS := atlas
PYTHON_INCLUDE := env3.6/include/python3.6m env3.6/lib/python3.6/site-packages/numpy/core/include
PYTHON_LIB := /usr/lib
INCLUDE_DIRS :=  /usr/local/include /usr/include/hdf5/serial
LIBRARY_DIRS :=  /usr/local/lib /usr/lib/x86_64-linux-gnu/hdf5/serial
PYTHON_LIBRARIES := boost_python-py36 python3.6m
BUILD_DIR := build
DISTRIBUTE_DIR := distribute
TEST_GPUID := 0
Q ?= @
EOF

make all 
make pycaffe

echo "Caffe Build Successful"
export PYTHONPATH=./python

python -c "import caffe; print(dir(caffe))"
#python -c 'import sys; sys.path.append("python"); import caffe; print(dir(caffe)); print("Installation Successful.");'

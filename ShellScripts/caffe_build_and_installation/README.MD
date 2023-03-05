                                ReadMe
################# Build Caffe for CPU on Ubuntu 18.04##################
Create virtualenv by using following command. 

1. virtualenv -p python3.6 env3.6
2. chmod +x build_caffe.sh
3. git clone https://github.com/BVLC/caffe.git
3. Copy "build_caffe.sh" in caffe-master.
3. ./build_caffe.sh

Above Commands will build caffe by automatically installing 
all dependencies.

4. To run caffe python API, type 

source env3.6/bin/activate
export PYTHONPATH=./python

$python
>>>import caffe
>>>print(dir(caffe))
######################################################################

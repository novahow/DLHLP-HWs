sudo apt update
sudo apt-get install tmux

pip install -e fairseq
# get into fairseq/modules/layer_norm.py
# and comment out line 31 and 32

pip install --upgrade numpy

wget https://dl.fbaipublicfiles.com/fairseq/models/xglm/xglm.7.5B.tar.gz
tar zxvf xglm.7.5B.tar.gz 


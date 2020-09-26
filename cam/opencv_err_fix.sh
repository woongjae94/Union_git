sudo apt-get -y install libgtk2.0-dev
sudo apt-get -y install pkg-config
conda remove opencv
conda update conda
conda install --channel menpo opencv
pip install opencv-contrib-python
sudo apt-get install libopencv-dev python-opencv

# 출처 : https://light-tree.tistory.com/150
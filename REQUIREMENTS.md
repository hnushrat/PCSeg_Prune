gcc version to 7


Add "deb [arch=amd64] http://archive.ubuntu.com/ubuntu focal main universe" to sudo nano  /etc/apt/sources.list    


sudo apt update
sudo apt install g++-7 gcc-7


Setting priority to gcc7 instead of your current gcc version, for me it was gcc11
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11 --slave /usr/bin/gcov gcov /usr/bin/gcov-11

cd /usr/bin

sudo rm x86_64-linux-gnu-gcc
sudo ln -sf gcc-7 x86_64-linux-gnu-gcc

sudo rm x86_64-linux-gnu-cpp

sudo ln -sf cpp-7 x86_64-linux-gnu-cpp
sudo rm x86_64-linux-gnu-g++
sudo ln -sf g++-7 x86_64-linux-gnu-g++

sudo apt-get install python3-dev
sudo apt install ninja-build

sudo apt-get install libsparsehash-dev


sudo apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

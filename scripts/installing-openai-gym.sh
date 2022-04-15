#On Ubuntu 18.04:

sudo apt install -y \
python3-dev \
zlib1g-dev \
libjpeg-dev \
cmake \
swig \
python3-opengl \
libboost-all-dev \
libsdl2-dev \
libosmesa6-dev \
patchelf \
ffmpeg \
xvfb

#python3-pyglet \

# After running the preceding command:
# git clone https://github.com/openai/gym.git
git submodule update --init --recursive

cd external/gym
pip install -e .

#Certain Gym environments also require the installation of pybox2d :
# git clone https://github.com/pybox2d/pybox2d
cd external/pybox2d
pip install -e .
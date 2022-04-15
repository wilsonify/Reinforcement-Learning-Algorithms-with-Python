#OpenAI Gym installation
#=====

# On Ubuntu 18.04
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
#python-pyglet \

# Then:
# git clone https://github.com/openai/gym.git
pip install -e 'gym[all]'

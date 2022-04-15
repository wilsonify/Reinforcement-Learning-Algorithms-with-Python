OpenAI Gym installation
=====

# On Ubuntu 18.04
```
sudo apt install -y python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb 
```

# Then:
```
git clone https://github.com/openai/gym.git 
cd gym
pip install -e '.[all]'
```

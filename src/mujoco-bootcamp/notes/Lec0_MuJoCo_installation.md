# MuJoCo installation

1. Download mujoco200 appropriate to your system: win/
    mac/linux Link: https://roboti.us/download.html
2. Download mjkey.txt Link: https://roboti.us/license.html
3. Unzip 1 in appropriate location on you computer (e.g.,
    Documents). Drop the mjkey.txt in the folder “bin”
    folder.

```
File structure in mujoco
```

# MuJoCo installation

4. Linux/Mac: Install make/gcc/make; Win: Install Visual
    studio installer then install “Desktop Development with
    C++). See the instructions on tiny.cc/mujoco
5. Recommended: Install Atom (atom.io) for editing C/xml
    files


# MuJoCo (checking installation)

1. In the shell*, navigate to “sample” folder. Type make. If there
    are no errors, you are all set.
2. In the shell*, navigate to “bin” folder. Type ./simulate ../model/
    arms26.xml (unix) or simulate ../model/arm26.xml. If you see
    a moving arm, this confirms you are ready to start working
    in MuJoCO.

*shell. For Linux/Mac that is the program “terminal”. For Win
that would be a x64 shell obtained as follows. Start -> Visual
Studio -> x64 Native Tools Command prompt


# MuJoCo (File structure)

- bin: Executables
- sample: c/c++ code and make
- model: xml files
- Current framework: In sample, type make; In bin type ./

```
<executable> ../model/<mujoco_model_file.xml>
```

# MuJoCo (our workspace)

- We will follow an easier framework
- We will follow a different framework. Create a myproject

```
folder.
```
- We will put makefile, C code, xml and run them from

```
this folder (hopefully that is simpler to you).
```


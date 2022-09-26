Using template_pendulum.zip to get started

1. From tiny.cc/mujoco download template_pendulum.zip and unzip in
    myproject
2. Rename folder template_pendulum to hopper
3. Make these three changes
    1. main.c — line 28, change template_pendulum/ to hopper/
    2. makefile — change ROOT = template_writeData to ROOT =
       hopper also UNCOMMENT (del #) appropriate to your OS
    3. run_unix / run_win.bat change <template_pendulum> to
       <hopper>
4. In the *shell, navigate to hoppper and type ./run_unix (unix) or
    run_win (windows); *shell = terminal for mac/linux / x64 for win


##### Model (xml)

## x

### z

#### q 1^1

###### Torso

###### Leg

###### Hip joint Foot

###### Knee

###### Joint

###### World

###### Torso

###### Leg

###### Foot

###### Knee Joint: l

###### Hip Joint: q

###### Translation: x and z


##### Finite State Machine: States

###### Air 1 Stance 1 Stance 2 Air 2

###### One Step


##### Finite State Machine: Transitions

```
Air 1 Stance 1 Stance 2 Air 2
Foot touches
ground
```
```
vz torso >0 Foot leavesground^
```
```
vz torso <
```

##### Finite State Machine: Actions (Knee joint/Height control)

###### Air 1 Stance 1 Stance 2 Air 2

pos/vel servo
(position control)

```
pos servo
(spring)
```
```
pos servo
(spring)
```
```
pos/vel servo
(position control)
```
##### All servos on reference pos/vel of 0


##### Finite State Machine: Actions (Hip joint/Velocity control)

###### Air 1 Stance 1 Stance 2 Air 2

```
pos/vel ref = 0 pos/vel ref = 0 pos=-0.
vel = 0
```
```
pos/vel = 0
```
##### All servos on fixed gains for pos/vel servo



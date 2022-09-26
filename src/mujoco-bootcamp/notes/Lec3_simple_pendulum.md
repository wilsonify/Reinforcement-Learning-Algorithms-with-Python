# MuJoCo: control a simple pendulum (1)

Using template to get started

1. From tiny.cc/mujoco download template.zip and unzip in myproject
2. Rename folder template to control_pendulum
3. Rename hello.xml to pendulum.xml
4. Make these three changes
    1. main.c — line 13, change template/hello.xml to control_pendulum/
       pendulum.xml
    2. makefile — change ROOT = template to ROOT = control_pendulum
       also UNCOMMENT (remove #) appropriate to your OS
    3. run_unix OR run_win.bat change <template> to <control_pendulum>
5. In the *shell, navigate to control_pendulum and type ./run_unix (unix) or
    run_win (windows); *shell = terminal for mac/linux and x64 (visual studio)
    for win


# MuJoCo: control a simple pendulum (2)

## Modify pendulum.xml

## 1. Create free swinging pendulum

## 2. Add 3 actuators: torque control, position servo, velocity

## servo

## 3. Add sensors: position and velocity

## Modify main.c

## 4. Torque control with/without sensor noise

## 5. Position servo (spring)

## 6. Velocity servo (speed control)

## 7. Position and velocity servo (position control)



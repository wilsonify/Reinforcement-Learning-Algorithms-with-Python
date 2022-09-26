Overview

1. Create a doublependulum model in xml
2. Check energy balance of a free pendulum
3. Check equations of motion
4. Torque-based position control using three methods
    i) position-derivative (PD) control
    ii) (gravity + coriolis forces) + PD control
    iii) feedback linearization control
5. Writing data file from MuJoCo and plotting in MATLAB


Using template_writeData.zip to get started

1. From tiny.cc/mujoco download template_writeData.zip and unzip in
    myproject
2. Rename folder template to dbpendulum
3. Rename pendulum.xml to doublependulum.xml
4. Make these three changes
    1. main.c — line 28, change template_writeData/ to dbpendulum/ and
       pendulum.xml to doublependulum.xml
    2. makefile — change ROOT = template_writeData to ROOT =
       dbpendulum also UNCOMMENT (remove #) appropriate to your OS
    3. run_unix / run_win.bat change <template_writeData> to <dbpendulum>
5. In the *shell, navigate to dbpendulum and type ./run_unix (unix) or run_win
    (windows); *shell = terminal for mac/linux and x64 (visual studio) for win


Main.c
1) Check energy

- mj_energyPos(m,d) & mj_energyVel(m,d);


Main.c
2) Check equations of motion: M qddot + C + G = tau

- M is mass matrix 2x 2
- qddot is acceleration, 2 x 1
- C is coriolis forces, 2x
- G is gravitational force, 2x 1
- tau is external torque, 2x


Main.c
2) Check equations of motion: M qddot + C + G = tau
MuJoCo equations of motion:
M qacc + qfrc_bias = qfrc_applied + ctrl

- qfrc_bias = C + G
- tau can be qfrc_applied OR ctrl
- qfrc_applied is always available (generalized force)
- ctrl is available on if an actuator is defined


Main.c
Equations: M qddot + f = tau where f = C + G
3) Controllers
i) Proportional-Derivative control
tau = -Kp*(q-q_ref) - Kd*qdot
ii) (gravity + coriolis forces) + PD control
tau = f -Kp*(q-q_ref) - Kd*qdot
iii) Feedback linearization
tau = M( -Kp*(q-q_ref) - Kd*qdot ) + f



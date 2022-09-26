# Using template_pendulum2.zip to get starte1. From tiny.cc/mujocoMuJoCo: 2D Biped (1) download template_pendulum2.zipd and

2.3. unzip in myprojecRename folder Make these three changes1. main.c — line 28, change template_pendulum2t (^) template_pendulum2 to biped / to biped/
2.3. make= bipedrun_unix / run_win.bat change <biped>file — change ROOT = template_writeData to also UNCOMMENT (del #) appropriate to your Otemplate_pendulum2ROOT > to <S

4. In the shell, navigate to biped and type ./run_unix (unix)


# Model (xml)MuJoCo: 2D Biped (2)

```
Hip jointLeg2 Foot
Knee joint 1
```
World Leg 1
Knee Joint: l1Foot 1 Hip Joint: q

## Un-actuated: x, z, q1 zq 1 x

## Knee Joint: l2Foot 2Leg 2 Leg1 Foot1 q^2^1 Knee joint 2^2


# State EstimationMuJoCo: 2D Biped (3)

```
leg
Absolute Angle (-) Absolute Angle (+)
Z-position of foot 2 Z-position of foot 1 leg
```
```
(+ leg 2 is in front)Relative angle^
```

## MuJoCo: Finite State Machine, Hip (4)leg2 swing

```
leg1 swing
```
```
foot2 on ground(abs_leg1 is -)leg1 is trailing foot1 on ground(abs_leg2 is -)leg2 is trailing
leg2 leg
```
**leg1 leg2** touchedjust

```
touchedjust^
```
```
q2 = 0.
q2 = -0.
```
```
leg2 leg
leg1 leg
```
```
q
q
```

### MuJoCo: Finite State Machine, Knee 1 (5)stance

```
retract
```
foot2 on ground(abs_leg1 is -)leg1 is trailing (^) (abs_leg1 > 0.1)leg1 is in front
**leg2 leg
leg1 leg2** touchedjust l1 = 0
l1=-0.5 **leg1 leg2** l1 = -0.25 l1=


### MuJoCo: Finite State Machine, Knee 2 (6)stance

```
retract
```
foot1 on ground(abs_leg1 is -)leg2 is trailing (^) (abs_leg2 > 0.1)leg2 is in front
**leg1 leg
leg2 leg1** touchedjust l2 = 0
l2=-0.5 **leg2 leg1** l2 = -0.25 l2=



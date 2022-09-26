Create XML

1. Create ball.xml
2. Readable xml: simulate > drop xml > Print model. See bin/MJMODEL.TXT
Create makefile, C, and executable
3. From tiny.cc/mujoco download template.zip and unzip in myproject
4. Rename template to projectile
5. Make these three changes
    1. main.c — line 13, change template/hello.xml to projectile/ball.xml
    2. makefile — change ROOT = template to ROOT = projectile also
       UNCOMMENT (remove #) appropriate to your OS
    3. run_unix OR run_win.bat change <template> to <projectile>
6. In the *shell, navigate to projectile and type ./run_main (unix) or run_win
    (windows); *shell = terminal for mac/linux and x64 (visual studio) for win


- Use the API reference for MuJoCo variables/functions;

## https://mujoco.readthedocs.io/en/latest/

## APIreference.html

- Bookmark this page.
- We will modify main.c using API reference.
    - m = mj_loadModel(...); //m = model
    - d = mj_makeData(m); //d = data
    - mj_step(d,m); //integrate for one time step


- Summary of commands in this section
    - Change the view; cam.azimuth and so on...
    - Change gravity: m-> opt.gravity
    - Show frames: opt.frame
    - Set init. position/velocity: d->qpos, d->qvel
    - Apply drag force: d->qfrc_applied



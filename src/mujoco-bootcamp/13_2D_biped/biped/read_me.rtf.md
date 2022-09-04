double M1\[nv1\*nv1\]={0};\
M1\[0\] = M\[0\]; M1\[1\] = M\[1\]; M1\[2\] = M\[2\];\
M1\[3\] = M\[6\]; M1\[4\] = M\[7\]; M1\[5\] = M\[8\];\
M1\[6\] = M\[12\]; M1\[7\] = M\[13\]; M1\[8\] = M\[14\];\
SLIDES: Explain Hybrid system and then modify template_pendulum\
\
\
SLIDES: Create floating base pendulum.\
\
\
main.c\
Turn on opt.flags\[mjVIS_JOINT\] = 1 ;\
\
pendulum.xml\
Turn to 0 degrees\
\<joint name=\"x\" type=\"slide\" pos=\"0 0 0.5\" axis = \"1 0 0\" /\>\
Change gravity=\"1 0 -9.81 to see it move.\
\
\<joint name=\"z\" type=\"slide\" pos=\"0 0 0.5\" axis = \"0 0 1\" /\>\
It will fall in the ground\
\
main.c\
d-\>qpos\[2\] = 0.5; falls in the ground\
\
pendulum.xml\
name the body "pole"\
\
Then\
\<equality\>\
\<connect body1=\'pole\' body2=\'world\' anchor=\'0 0 0.5\'/\>\
\</equality\>\
\
Now pendulum oscillates\
\
\
SLIDES: Explain physics of floating pendulum. Then get to verify the
equation of motion.\
\
main.c\
Check equations of motion\
\
int i,j;\
const int nv = 3;\
\
double M\[nv\*nv\]={0};\
mj_fullM(m, M, d-\>qM);//MJAPI void mj_fullM(const mjModel\* m, mjtNum\*
dst, const mjtNum\* M);\
\
double f\[nv\] = {0};\
f\[0\] = d-\>qfrc_bias\[0\];\
f\[1\] = d-\>qfrc_bias\[1\];\
f\[2\] = d-\>qfrc_bias\[2\];\
\
double qddot\[nv\] = {0};\
qddot\[0\] = d-\>qacc\[0\];\
qddot\[1\] = d-\>qacc\[1\];\
qddot\[2\] = d-\>qacc\[2\];\
\
double J\[3\*nv\] = {0};\
for (i=0;i\<3\*nv;i++)\
J\[i\]=d-\>efc_J\[i\];\
\
double lambda\[3\]={0};\
for (i=0;i\<3;i++)\
lambda\[i\] = d-\>efc_force\[i\];\
\
//printf(\" M\*qddot + f - J\'\*lambda = 0\\n\\n\");\
double M_qddot\[3\]={0};\
mju_mulMatVec(M_qddot,M,qddot,3,3);\
\
double J_lambda\[3\]={0};\
mju_mulMatTVec(J_lambda,J,lambda,3,3);\
\
double lhs\[3\] = {0};\
for (i=0;i\<3;i++)\
{\
lhs\[i\] = M_qddot\[i\]+f\[i\] - J_lambda\[i\];\
printf(\"%f \\n\",lhs\[i\]);\
}\
\
printf(\"Constraint residual (should be zero) = %f %f %f
\\n\\n\",d-\>efc_pos\[0\],d-\>efc_pos\[1\],d-\>efc_pos\[2\]);\
\
\
printf(\"\*\*\*\*\*\\n\");\
\
SLIDES: Slides on creating another pendulum and adding forces\
\
pendulum.xml\
Add another pendulum with 3 dofs\
\
\<body name=\"pole2\" pos=\"0 -1 2\" euler=\"0 0 0\"\>\
\<joint name=\"x2\" type=\"slide\" pos=\"0 0 0.5\" axis = \"1 0 0\" /\>\
\<joint name=\"z2\" type=\"slide\" pos=\"0 0 0.5\" axis = \"0 0 1\" /\>\
\<joint name=\"pin2\" type=\"hinge\" pos=\"0 0 0.5\" axis=\"0 -1 0\"
/\>\
\<geom type=\"cylinder\" size=\".05 .5\" rgba=\".9 0 0 1\"
mass=\"1\"/\>\
\</body\>\
\
main.c\
Change nv to 6 else you will get an error.\
\
double M1\[nv1\*nv1\]={0};\
M1\[0\] = M\[0\]; M1\[1\] = M\[1\]; M1\[2\] = M\[2\];\
M1\[3\] = M\[6\]; M1\[4\] = M\[7\]; M1\[5\] = M\[8\];\
M1\[6\] = M\[12\]; M1\[7\] = M\[13\]; M1\[8\] = M\[14\];\
\
double J1\[3\*nv1\]={0};\
J1\[0\] = J\[0\]; J1\[1\] = J\[1\]; J1\[2\] = J\[2\];\
J1\[3\] = J\[6\]; J1\[4\] = J\[7\]; J1\[5\] = J\[8\];\
J1\[6\] = J\[12\]; J1\[7\] = J\[13\]; J1\[8\] = J\[14\];\
\
Change M to M1 and J to J1 and check equations\
lhs\[i\] = M_qddot\[i\]+f\[i\] - J_lambda\[i\];\
\
Now compute forces for the 2nd pendulum and apply. Don't forget to start
both pendulums from same position.\
d-\>qfrc_applied\[3\] = J_lambda\[0\];\
d-\>qfrc_applied\[4\] = J_lambda\[1\];\
d-\>qfrc_applied\[5\] = J_lambda\[2\];\
\
SLIDES: Now use FSM\
\
\
Create fsm\
\
int fsm;\
\
#define fsm_swing 0\
#define fsm_free 1\
\
init_controller\
fsm = fsm_swing;\
\
double angle_deg = d-\>qpos\[5\]\*(180/3.14);\
angle_deg = constrainAngle(angle_deg);\
double angle_rad = angle_deg\*(3.14/180);\
if (fsm == fsm_swing && angle_rad\>1.2 && angle_rad\<2 &&
d-\>qvel\[5\]\>5)\
fsm = fsm_free;\
\
if (fsm==fsm_swing)\
{\
d-\>qfrc_applied\[2\] = 0.1\*d-\>time;\
\
d-\>qfrc_applied\[3\] = J_lambda\[0\];\
d-\>qfrc_applied\[4\] = J_lambda\[1\];\
d-\>qfrc_applied\[5\] = J_lambda\[2\]+d-\>qfrc_applied\[2\];\
}\
if (fsm==fsm_free)\
{\
d-\>qfrc_applied\[2\] = -0.5\*d-\>qvel\[2\];\
d-\>qfrc_applied\[3\] = 0;\
d-\>qfrc_applied\[4\] = 0;\
d-\>qfrc_applied\[5\] = 0;\
}\
\
double constrainAngle(double x)\
{\
x = fmod(x + 180,360);\
if (x \< 0)\
x += 360;\
return x - 180;\
}\
\
d-\>qpos\[2\] = -2;\
d-\>qpos\[5\] = d-\>qpos\[2\];\
\
\
\
\
\
\"Hernandez Hinojosa, Ernesto A\" \<eherna95@uic.edu\>, Salvador
Echeveste \<sechev6@uic.edu\>, \"Bittler, James R\"
\<jbittler@uic.edu\>, Jonathan Garcia \<jgarc248@uic.edu\>, Subramanian
Ramasamy \<sramas21@uic.edu\>, \"Torres, Daniel\" \<dtorre38@uic.edu\>,
J K \<robotics68@gmail.com\>, Mohammad Safwan Mondal
\<mmonda4@uic.edu\>, \"Yang, Chun-Ming\" \<jyang241@uic.edu\>, \"Mittal,
Tanmay\" \<tmitta3@uic.edu\>, Giuseppe Cerruto \<gcerru2@uic.edu\>\
\

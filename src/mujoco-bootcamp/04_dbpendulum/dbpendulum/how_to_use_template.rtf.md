

1) Copy paste the template_writeFile file and rename suitably (say "pendulum")
2) Copy paste the appropriate .xml files in this folder (say "simple.xml")
3) Make the following changes to the contents in this folder
   1) main.c: the filename should be set to the correct path and xml file. 
      (e.g., char filename\[\] = \"../myproject/pendulum/simple.xml\";
   2) makefile: There are two changes. Change ROOT = pendulum and
      uncomment appropriate COMMON/LIBS/CC for the operating system
   3) run_unix or run_win.bat: Change from template to "pendulum"\
4) Navigate to the folder in cmd/terminal and type ./run_unix (for
mac/linux) and run_win (for windows)\

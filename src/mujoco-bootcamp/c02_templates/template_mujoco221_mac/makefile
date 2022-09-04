
#MAC
GLFWROOT?=/usr/local/Cellar/glfw/3.3.8
MUJOCOPATH?=/Applications/MuJoCo.app/Contents/Frameworks

CFLAGS=-O2 -F$(MUJOCOPATH) -I$(GLFWROOT)/include -pthread
ALLFLAGS=$(CFLAGS) -L$(GLFWROOT)/lib -Wl,-rpath,$(MUJOCOPATH)
LIBS = -framework mujoco -lglfw
CC = gcc

ROOT = main

all:
	$(CC) $(ALLFLAGS) main.c $(LIBS) -o $(ROOT)

main.o:
	$(CC) $(COMMON) -c main.c

clean:
	rm *.o 
	

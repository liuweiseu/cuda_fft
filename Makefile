CC	= gcc

DEF		= 
TARGET  = fast_gpu

INC		= fast_gpu.h
SRC		= fast_gpu.cpp

FLAG  	= -I./fast_gpu_lib \
          -L./fast_gpu_lib \
		  -lfastgpu -lm

SUB_DIR = fast_gpu_lib

${TARGET}: ${SRC} ${SUB_DIR}
	${CC} ${SRC} ${DEF} -o $@  ${FLAG} 

${SUB_DIR}: ECHO
	make -C $@

ECHO:
	@echo Going to compile .so in ${SUB_DIR}...

.PHONY: clean install
install:
	make -C fast_gpu_lib install
clean:
	rm ${TARGET}
	make -C fast_gpu_lib clean
# roly Makefile 

help:
	@echo "Available tasks:"
	@echo "help   -->  This help page"
	@echo "build  -->  Build the Cython extension modules"
	@echo "clean  -->  Remove all the build files for a fresh start"

all: clean build

build:
	python setup.py build_ext --inplace

# Phony targets for cleanup and similar uses

.PHONY: clean
clean:
	rm -rf *.so build doubleheap.c doubleheap2.c doubleheap3.c linkedlist.c

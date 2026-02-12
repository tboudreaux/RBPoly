default: all

all: run

run: main.cpp
	clang++ -O3 -o run main.cpp -lmfem -std=c++23

run_debug: main.cpp
	clang++ -g -O0 -o run_debug main.cpp -lmfem -std=c++23

clean:
	rm -f run run_debug

@.PHONY: all run run_debug clean

test: run
	./run

debug: run_debug
	lldb run_debug

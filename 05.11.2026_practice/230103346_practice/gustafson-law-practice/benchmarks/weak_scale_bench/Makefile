# Linux:  make                      (needs g++ with OpenMP)
# macOS:  brew install libomp && make   (Apple clang has no built-in OpenMP; Makefile uses Homebrew's libomp)
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  LIBOMP := $(shell brew --prefix libomp 2>/dev/null)
  CXX      ?= clang++
  CXXFLAGS := -O2 -std=c++17 -Xpreprocessor -fopenmp -I$(LIBOMP)/include
  LDFLAGS  := -L$(LIBOMP)/lib -lomp
else
  CXX      ?= g++
  CXXFLAGS := -O2 -std=c++17 -fopenmp
  LDFLAGS  :=
endif
weak_scale: weak_scale.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
clean:
	rm -f weak_scale

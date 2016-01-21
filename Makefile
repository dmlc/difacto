CXX = g++
DEPS_PATH = $(shell pwd)/deps

INCPATH = -I./src -I./include -I./dmlc-core/include -I./ps-lite/include -I./dmlc-core/src -I$(DEPS_PATH)/include
PROTOC = ${DEPS_PATH}/bin/protoc
CFLAGS = -std=c++11 -fopenmp -fPIC -O3 -ggdb -Wall -finline-functions $(INCPATH) -DDMLC_LOG_FATAL_THROW=0
# LDFLAGS += $(addprefix $(DEPS_PATH)/lib/, libprotobuf.a libzmq.a)

OBJS = $(addprefix build/, loss/loss.o \
updater.o sgd/sgd_updater.o \
learner.o \
store/store.o \
tracker/tracker.o \
reporter/reporter.o \
common/localizer.o data/batch_iter.o )

DMLC_DEPS = dmlc-core/libdmlc.a

all: build/difacto test

clean:
	rm -rf build
	make -C dmlc-core clean
	make -C ps-lite clean

lint:
	python2 dmlc-core/scripts/lint.py difacto all include src

# include ps-lite/make/deps.mk

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(INCPATH) -std=c++0x -MM -MT build/$*.o $< >build/$*.d
	$(CXX) $(CFLAGS) -c $< -o $@

build/libdifacto.a: $(OBJS)
	ar crv $@ $(filter %.o, $?)

build/difacto: build/main.o build/libdifacto.a $(DMLC_DEPS)
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS)

dmlc-core/libdmlc.a:
	$(MAKE) -C dmlc-core libdmlc.a DEPS_PATH=$(DEPS_PATH) CXX=$(CXX)

include tests/cpp/test.mk
cpp-test: $(CPPTEST)

test: cpp-test

-include build/*.d
-include build/*/*.d

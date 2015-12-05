CXX = g++
DEPS_PATH = $(shell pwd)/deps

INCPATH = -I./src -I./include -I./dmlc-core/include -I./ps-lite/include -I./dmlc-core/src -I$(DEPS_PATH)/include
PROTOC = ${DEPS_PATH}/bin/protoc
CFLAGS = -std=c++11 -fopenmp -fPIC -O3 -ggdb -Wall -finline-functions $(INCPATH)
# LDFLAGS += $(addprefix $(DEPS_PATH)/lib/, libprotobuf.a libzmq.a)

OBJS = $(addprefix build/, job.o difacto.o loss/loss.o model/model.o model_sync.o \
	common/localizer.o data/batch_iter.o model/sgd.o)

DMLC_DEPS = dmlc-core/libdmlc.a

all: build/difacto

clean:
	rm -rf build

lint:
	python2 dmlc-core/scripts/lint.py difacto all include src

# include ps-lite/make/deps.mk

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(INCPATH) -std=c++0x -MM -MT build/$*.o $< >build/$*.d
	$(CXX) $(CFLAGS) -c $< -o $@

build/difacto.a: $(OBJS)
	ar crv $@ $(filter %.o, $?)

build/difacto: build/main.o build/difacto.a $(DMLC_DEPS)
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS)

dmlc-core/libdmlc.a:
	$(MAKE) -C dmlc-core libdmlc.a DEPS_PATH=$(DEPS_PATH) CXX=$(CXX)

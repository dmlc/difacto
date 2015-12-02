CXX = g++
DEPS_PATH = $(shell pwd)/deps

INCPATH = -I./src -I./include -I./dmlc-core/include -I./ps-lite/include -I./dmlc-core/src -I$(DEPS_PATH)/include
PROTOC = ${DEPS_PATH}/bin/protoc
CFLAGS = -std=c++11 -fPIC -O3 -ggdb -Wall -finline-functions $(INCPATH)
# LDFLAGS += $(addprefix $(DEPS_PATH)/lib/, libprotobuf.a libzmq.a)

OBJS = $(addprefix build/, job.o difacto.o loss.o model.o model_sync.o common/localizer.o)

all: build/difacto

clean:
	rm -rf build
	find src -name "*.pb.[ch]*" -delete

lint:
	python2 dmlc-core/scripts/lint.py difacto all include src

include ps-lite/make/deps.mk

src/%.pb.cc src/%.pb.h : src/%.proto ${PROTOBUF}
	$(PROTOC) --cpp_out=./src --proto_path=./src $<

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(INCPATH) -std=c++0x -MM -MT build/$*.o $< >build/$*.d
	$(CXX) $(CFLAGS) -c $< -o $@

build/difacto.a: $(OBJS)
	ar crv $@ $(filter %.o, $?)

build/difacto: build/main.o build/difacto.a
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS)

GTEST_PATH = /usr

CPPTEST_SRC = $(wildcard tests/cpp/*_test.cc)
CPPTEST_OBJ = $(patsubst tests/cpp/%_test.cc, build/tests/%_test.o, $(CPPTEST_SRC))

build/tests/%.o : tests/cpp/%.cc ${DEPS}
	@mkdir -p $(@D)
	$(CXX) $(INCPATH) -std=c++0x -MM -MT build/tests/$*.o $< >build/tests/$*.d
	$(CXX) $(CFLAGS) -c $< -o $@

build/difacto_tests: $(CPPTEST_OBJ) build/tests/main.o build/libdifacto.a $(DMLC_DEPS)
	$(CXX) $(CFLAGS) -I$(GTEST_PATH)/include -o $@ $^ $(LDFLAGS) -L$(GTEST_PATH)/lib -lgtest

CPPPERF_SRC = $(wildcard tests/cpp/*_perf.cc)
CPPPERF = $(patsubst tests/cpp/%_perf.cc, build/%_perf, $(CPPTEST_SRC))


build/%_perf : tests/cpp/%_perf.cc build/libdifacto.a $(DMLC_DEPS) ${DEPS}
	$(CXX) -std=c++0x $(CFLAGS) -MM -MT $@ $< >$@.d
	$(CXX) -std=c++0x $(CFLAGS) -I$(GTEST_PATH)/include -o $@ $(filter %.cc %.a, $^) $(LDFLAGS)

cpp-perf: $(CPPPERF)

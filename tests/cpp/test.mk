GTEST_PATH = /usr

CPPTEST_SRC = $(wildcard tests/cpp/*_test.cc)
CPPTEST = $(patsubst tests/cpp/%_test.cc, build/%_test, $(CPPTEST_SRC))

build/%_test : tests/cpp/%_test.cc build/libdifacto.a $(DMLC_DEPS)
	$(CXX) -std=c++0x $(CFLAGS) -MM -MT $@ $< >$@.d
	$(CXX) -std=c++0x $(CFLAGS) -I$(GTEST_PATH)/include -o $@ $(filter %.cc %.a, $^) $(LDFLAGS) -L$(GTEST_PATH)/lib -lgtest -lgtest_main

CPPPERF_SRC = $(wildcard tests/cpp/*_perf.cc)
CPPPERF = $(patsubst tests/cpp/%_perf.cc, build/%_perf, $(CPPTEST_SRC))

build/%_perf : tests/cpp/%_perf.cc build/libdifacto.a $(DMLC_DEPS)
	$(CXX) -std=c++0x $(CFLAGS) -MM -MT $@ $< >$@.d
	$(CXX) -std=c++0x $(CFLAGS) -I$(GTEST_PATH)/include -o $@ $(filter %.cc %.a, $^) $(LDFLAGS)

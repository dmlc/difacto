CPPTEST_SRC = $(wildcard tests/cpp/*_test.cc)
CPPTEST = $(patsubst tests/cpp/%_test.cc, build/%_test, $(CPPTEST_SRC))

GTEST_PATH = /usr

build/%_test : tests/cpp/%_test.cc build/libdifacto.a $(DMLC_DEPS)
	$(CXX) -std=c++0x $(CFLAGS) -MM -MT $@ $< >$@.d
	$(CXX) -std=c++0x $(CFLAGS) -I$(GTEST_PATH)/include -o $@ $(filter %.cc %.a, $^) $(LDFLAGS) -L$(GTEST_PATH)/lib -lgtest -lgtest_main

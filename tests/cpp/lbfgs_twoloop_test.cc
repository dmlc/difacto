#include <gtest/gtest.h>
#include "lbfgs/lbfgs_twoloop.h"
#include "./utils.h"

using namespace difacto;
using namespace difacto::lbfgs;

/**
 * \brief a reference implementation
 */
void TwoloopRefer(const std::vector<SArray<real_t>>& s,
                  const std::vector<SArray<real_t>>& y,
                  const SArray<real_t>& g,
                  SArray<real_t>* p) {
  int m = s.size();
  std::vector<double> alpha(m);
  size_t n = g.size();
  p->resize(n); memset(p->data(), 0, n*sizeof(real_t));

  Add(-1, g, p);
  for (int i = m-1; i >= 0; i--) {
    alpha[i] = Inner(s[i], *p) / Inner(s[i], y[i]);
    Add(-alpha[i], y[i], p);
  }

  if (m > 0) {
    double x = Inner(s[m-1], y[m-1])/Inner(y[m-1], y[m-1])-1;
    Add(x, *p, p);
  }

  for (int i = 0; i < m; ++i) {
    double beta = Inner(y[i], *p) / Inner(s[i], y[i]);
    Add(alpha[i]-beta, s[i], p);
  }
}

TEST(Twoloop, naive) {

  SArray<real_t> g = {1, 2};
  std::vector<SArray<real_t>> s = {{2, 3}}, y = {{3, 4}};
  SArray<real_t> p0, p1;

  TwoloopRefer(s, y, g, &p0);

  Twoloop two;
  std::vector<real_t> B;
  two.CalcIncreB(s, y, g, &B);
  two.ApplyIncreB(B);
  two.CalcDirection(s, y, g, &p1);

  real_t a = (54.0-202)/25/9;
  real_t b = (-36.0-303)/25/9;

  EXPECT_LE(abs(p0[0] - a), 1e-5);
  EXPECT_LE(abs(p0[1] - b), 1e-5);
  EXPECT_LE(abs(p1[0] - a), 1e-5);
  EXPECT_LE(abs(p1[1] - b), 1e-5);
}

TEST(Twoloop, basic) {
  int m = 4;
  int n = 100;
  std::vector<SArray<real_t>> s, y;
  SArray<real_t> g;
  SArray<real_t> p0, p1;
  Twoloop two;
  for (int k = 0; k < 10; ++k) {
    gen_vals(n, -1, 1, &g);

    if (static_cast<int>(s.size()) == m-1) {
      s.erase(s.begin());
      y.erase(y.begin());
    }
    SArray<real_t> a, b;
    gen_vals(n, -1, 1, &a);
    gen_vals(n, -1, 1, &b);
    s.push_back(a);
    y.push_back(b);

    TwoloopRefer(s, y, g, &p0);

    std::vector<real_t> B;
    two.CalcIncreB(s, y, g, &B);
    two.ApplyIncreB(B);
    two.CalcDirection(s, y, g, &p1);

    EXPECT_LE(abs(norm2(p0) - norm2(p1)) / norm2(p1), 5e-6);
  }
}

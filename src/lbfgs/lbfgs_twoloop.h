#ifndef _VECTOR_FREE_H_
#define _VECTOR_FREE_H_
#include "./lbfgs_utils.h"
namespace difacto {
namespace lbfgs {
/**
 * \brief compute direction p given s, y, ∇f(w)
 *
 * we used the vector-free version, reference paper:
 *
 * chen et al, large-scale l-bfgs using mapreduce, nips, 2015
 */
class Twoloop {
 public:
  void CalcIncreB(const std::vector<SArray<real_t>>& s,
                  const std::vector<SArray<real_t>>& y,
                  const SArray<real_t>& grad,
                  std::vector<real_t>* incr_B) {
    CHECK_EQ(s.size(), y.size());
    int m = static_cast<int>(s.size());
    incr_B->resize(6*m+1);
    for (int i = 0; i < m; ++i) {
      (*incr_B)[i    ] = Inner(s.back(), s[i]);
      (*incr_B)[i+  m] = Inner(s.back(), y[i]);
      (*incr_B)[i+2*m] = Inner(y.back(), s[i]);
      (*incr_B)[i+3*m] = Inner(y.back(), y[i]);
      (*incr_B)[i+4*m] = Inner(grad,     s[i]);
      (*incr_B)[i+5*m] = Inner(grad,     y[i]);
    }
    (*incr_B)[6*m] = Inner(grad, grad);
  }

  void ApplyIncreB(const std::vector<real_t>& incr_B) {
    int m = static_cast<int>((incr_B.size() - 1) / 6);
    CHECK(m == m_+1 || m == m_);
    std::vector<std::vector<double>> B;
    for (int i = 0; i < 2*m+1; ++i) {
      std::vector<double> b(2*m+1);
      if (i < m - 1) {
        const auto& old = B_[i+(m==m_?1:0)];
        for (int j = 0; j <= i; ++j) b[j] = old[j+(m==m_?1:0)];
      } else if (i == m-1) {
        for (int j = 0; j <= i; ++j) b[j] = incr_B[j];
      } else if (i < 2*m-1) {
        const auto& old = B_[i+(m==m_?1:-1)];
        for (int j = 0; j < m; ++j) b[j] = old[j+(m==m_?1:0)];
        b[m-1] = incr_B[i];
        for (int j = m; j <= i; ++j) b[j] = old[j+(m==m_?1:-1)];
      } else if (i == 2*m-1) {
        for (int j = 0; j < 2*m; ++j) b[j] = incr_B[2*m+j];
      } else {
        for (int j = 0; j < 2*m+1; ++j) b[j] = incr_B[4*m+j];
      }
      B.push_back(b);
    }
    for (int i = 0; i < 2*m+1; ++i) {
      for (int j = 0; j < i; ++j) {
        B[j][i] = B[i][j];
      }
    }
    B_ = B;
    m_ = m;
  }

  /**
   * \brief compute p based on s, y, ∇f(w)
   *
   * One need to call CalcIncreB and ApplyIncrB first
   *
   * @param s [s(k-m), ..., s(k-1)]
   * @param y [y(k-m), ..., y(k-m)]
   * @param grad ∇f(w(k))
   * @param p the l-bfgs direction
   */
  void CalcDirection(const std::vector<SArray<real_t>>& s,
                     const std::vector<SArray<real_t>>& y,
                     const SArray<real_t>& grad,
                     SArray<real_t>* p) {
    CHECK_EQ(s.size(), static_cast<size_t>(m_));
    CHECK_EQ(y.size(), static_cast<size_t>(m_));
    size_t n = grad.size();
    p->resize(n); memset(p->data(), 0, n*sizeof(real_t));

    std::vector<double> delta; CalcDelta(&delta);
    for (int i = 0; i < m_; ++i) Add(delta[i], s[i], p);
    for (int i = 0; i < m_; ++i) Add(delta[i+m_], y[i], p);
    Add(delta[2*m_], grad, p);
  }

 private:
  void CalcDelta(std::vector<double>* delta) {
    delta->resize(2*m_+1);
    double* d = delta->data(); d[2*m_] = -1;

    std::vector<double> alpha(m_);
    for (int i = m_ - 1; i >= 0; --i) {
      for (int l = 0; l < 2*m_+1; ++l) {
        alpha[i] += d[l] * B_[l][i];
      }
      alpha[i] /= B_[i][m_+i] + 1e-10;
      d[m_+i] -= alpha[i];
    }

    for (int i = 0; i < 2*m_+1; ++i) {
      d[i] *= B_[m_-1][2*m_-1] / (B_[2*m_-1][2*m_-1] + 1e-10);
    }

    for (int i = 0; i < m_; ++i) {
      double beta = 0;
      for (int l = 0; l < 2*m_+1; ++l) {
        beta += d[l] * B_[m_+i][l];
      }
      beta /= B_[i][m_+i] + 1e-10;
      d[i] += alpha[i] - beta;
    }
  }

  int m_ = 0;
  int nthreads_ = DEFAULT_NTHREADS;
  /** \brief B_[i][j] = <b[i], b[j]> */
  std::vector<std::vector<double>> B_;

};
}  // namespace lbfgs
}  // namespace difacto
#endif  // _VECTOR_FREE_H_

#ifndef _VECTOR_FREE_H_
#define _VECTOR_FREE_H_
namespace difacto {
namespace lbfgs {
/**
 * \brief vector-free lbfgs
 * reference paper:
 * chen et al, large-scale l-bfgs using mapreduce, nips, 2015
 */
class Twoloop {
 public:
  Twoloop(int nthreads) : nthreads_(nthreads) { }
  void CalcIncreB(const std::vector<SArray<real_t>>& s,
                  const std::vector<SArray<real_t>>& y,
                  const SArray<real_t>& grad;
                  std::vector<real_t>* incr_B) {

  }

  void ApplyIncrB(const std::vector<real_t>& incr_B) {

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
                     std::vector<real_t>* p) {
    CHECK_EQ(s.size(), m_);
    CHECK_EQ(y.size(), m_);
    size_t n = grad.size();
    p->resize(n); memset(p.data(), 0, n*sizeof(real_t));

    std::vector<double> delta; CalcDelta(&delta);
    for (int i = 0; i < m_; ++i) Add(s[i], delta[i], p);
    for (int i = 0; i < m_; ++i) Add(y[i], delta[i+m_], p);
    Add(grad, delta[2*m_], p);
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
      alpha[i] /= B_[i][m_+i];
      d[m_+i] -= alpha[i];
    }

    for (int i = 0; i < 2*m_+1; ++i) {
      d[i] *= B_[m_][2*m_] / B_[2*m_][2*m_];
    }

    for (int i = 0; i < m_; ++i) {
      double beta = 0;
      for (int l = 0; l < 2*m_+1; ++l) {
        beta += d[l] * B_[m_+i][l];
      }
      beta /= B_[i][m_+i];
      d[i] += alpha[i] - beta;
    }
  }

  /**
   * \brief b += a * x
   */
  real_t Add(const SArray<real_t>& a, real_t x, SArray<real_t>* b) {
    CHECK_EQ(a.size(), b->size());
    real_t const *ap = a.data();
    real_t *bp = b.data();
#pragma omp parallel for num_threads(nthreads_)
    for (size_t i = 0; i < a.size(); ++i) bp[i] += x * ap[i];
  }

  int m_ = 0;
  // B_[i][j] = <b[i], b[j]>
  std::vector<std::vector<double>> B_;

};
}  // namespace lbfgs
}  // namespace difacto
#endif  // _VECTOR_FREE_H_

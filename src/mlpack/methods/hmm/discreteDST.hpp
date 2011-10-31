#ifndef __MLPACK_METHODS_HMM_DISCRETE_DISTRIBUTION_HPP
#define __MLPACK_METHODS_HMM_DISCRETE_DISTRIBUTION_HPP

#include <mlpack/core.h>

namespace mlpack {
namespace hmm {

class DiscreteDST {
 private:
  arma::vec p;
  arma::vec acc_p;

 public:
  void Init(int n = 2);

  void generate(int* v);

  double get(int i) { return p[i]; }

  void set(const arma::vec& p_) { p = p; }

  void start_accumulate() { acc_p.zeros(); }

  void accumulate(int i, double v) { acc_p[i] += v; }

  void end_accumulate();
};

}; // namespace hmm
}; // namespace mlpack

#endif // __MLPACK_METHODS_HMM_DISCRETE_DISTRIBUTINO_HPP

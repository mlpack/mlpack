#ifndef FASTLIB_DISCRETE_DISTRIBUTION_H
#define FASTLIB_DISCRETE_DISTRIBUTION_H

#include <fastlib/fastlib.h>
#include <armadillo>

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

#endif

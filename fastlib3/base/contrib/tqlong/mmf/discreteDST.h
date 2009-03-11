#ifndef FASTLIB_DISCRETE_DISTRIBUTION_H
#define FASTLIB_DISCRETE_DISTRIBUTION_H
#include "fastlib/fastlib.h"
class DiscreteDST {
  Vector p;
  Vector ACC_p;
 public:
  void Init(int N = 2);
  void generate(int* v);
  double get(int i) { return p[i]; }
  void set(const Vector& p_) { p.CopyValues(p_); }

  void start_accumulate() { ACC_p.SetZero(); }
  void accumulate(int i, double v) { ACC_p[i]+=v; }
  void end_accumulate();
};
#endif

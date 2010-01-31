#include "fastlib/fastlib.h"
#include "discreteDST.h"
#include "support.h"

void DiscreteDST::Init(int N) {
  p.Init(N);
  ACC_p.Init(N);
  double s = 1;
  for (int i = 0; i < N-1; i++) {
    p[i] = RAND_UNIFORM(s*0.2,s*0.8);
    s -= p[i];
  }
  p[N-1] = s;
}

void DiscreteDST::generate(int* v) {
  int N = p.length();
  double r = RAND_UNIFORM_01;
  double s = 0;
  for (int i = 0; i < N; i++) {
    s += p[i];
    if (s >= r) {
      *v = i;
      return;
    }
  }
  *v = N-1;
}

void DiscreteDST::end_accumulate() {
  int N = p.length();
  double s = 0;
  for (int i = 0; i < N; i++) s += ACC_p[i];
  if (s == 0) s = -INFINITY;
  for (int i = 0; i < N; i++) p[i] = ACC_p[i]/s;
}

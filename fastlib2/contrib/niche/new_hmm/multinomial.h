#include "fastlib/fastlib.h"

#ifndef MULTINOMIAL_H
#define MULTINOMIAL_H

class Multinomial {
 public:

  int n_dims_;
  int n_components_;

  Vector* p_;

  void Init(int n_dims_in) {
    Init(n_dims_in, 1);
  }

  void Init(int n_dims_in, int n_components_in) {
    n_dims_ = n_dims_in;
    n_components_ = n_components_in;
    
    p_ = new Vector();
    p_ -> Init(n_dims_);
  }
  
  void Init(const Vector &p_in) {
    p_ = new Vector();
    p_ -> Copy(p_in);

    n_dims_ = p_ -> length();
  }

  void CopyValues(const Multinomial &other) {
    p_ -> CopyValues(other.p());
  }

  int n_dims () const {
    return n_dims_;
  }

  const Vector p () const {
    return *p_;
  }

  void SetP(const Vector &p_in) {
    p_ ->CopyValues(p_in);
  }

  void RandomlyInitialize() {
    double sum = 0;
    for(int i = 0; i < n_dims_; i++) {
      (*p_)[i] = drand48();
      sum = sum + (*p_)[i];
    }

    for(int i = 0; i < n_dims_; i++) {
      (*p_)[i] = (*p_)[i] / sum;
    }
  }

  void PrintDebug(const char *name = "", FILE *stream = stderr) const {
    fprintf(stream, name);
    p_ -> PrintDebug("p");
  }

  ~Multinomial() {
    Destruct();
  }

  void Destruct() {
    delete p_;
  }

};
    
#endif /* MULTINOMIAL_H */

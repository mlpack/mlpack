#ifndef MULTINOMIAL_H
#define MULTINOMIAL_H

#include "fastlib/fastlib.h"
#include "la_utils.h"


class Multinomial {
 public:

  int n_dims_;
  int n_components_;

  Vector* p_;
  Vector* logp_;

 private:


  OBJECT_TRAVERSAL_ONLY(Multinomial) {
    OT_PTR(p_);
    OT_PTR(logp_);
  }


 public:

  void Init(int n_dims_in) {
    Init(n_dims_in, 0);
  }

  void Init(int n_dims_in, double min_variance_in) {
    Init(n_dims_in, min_variance_in, 1);
  }

  void Init(int n_dims_in, double min_variance_in, int n_components_in) {
    n_dims_ = n_dims_in;
    n_components_ = n_components_in;
    
    p_ = new Vector();
    p_ -> Init(n_dims_);

    logp_ = new Vector();
    logp_ -> Init(n_dims_);
  }
  
  void Init(const Vector &p_in) {
    p_ = new Vector();
    p_ -> Copy(p_in);

    n_dims_ = p_ -> length();

    logp_ = new Vector();
    logp_ -> Init(n_dims_);
    UpdateLogP();
  }

  void CopyValues(const Multinomial &other) {
    p_ -> CopyValues(other.p());
    logp_ -> CopyValues(other.logp());
  }

  int n_dims() const {
    return n_dims_;
  }

  const Vector p() const {
    return *p_;
  }
  
  const Vector logp() const {
    return *logp_;
  }

  void SetP(const Vector &p_in) {
    p_ ->CopyValues(p_in);
    
    UpdateLogP();
  }

  void UpdateLogP() {
    for(int i = 0; i < n_dims_; i++) {
      (*logp_)[i] = log((*p_)[i]);
    }
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

    UpdateLogP();
  }

  template<typename T>
  double PkthComponent(const GenVector<T> &xt, int component_num) {
    return Pdf(xt);
  }

  template<typename T>
    double LogPkthComponent(const GenVector<T> &xt, int component_num) {
    return LogPdf(xt);
  }

  template<typename T>
  double Pdf(const GenVector<T> &xt) {
    return (*p_)[xt[0]];
  }

  template<typename T>
  double LogPdf(const GenVector<T> &xt) {
    return (*logp_)[xt[0]];
  }

  // call the same update function for all HMMs
  void SetZero() {
    p_ -> SetZero();
  }
  
  template<typename T>
  void Accumulate(double weight, const GenVector<T> &example,
		  int component_num){
    (*p_)[example[0]] += weight;
  }
  
  void Normalize(double normalization_factor) {
    double sum = Sum(*p_);
    la::Scale(((double)1) / sum, p_);

    UpdateLogP();
  }

  void Normalize(double normalization_factor,
		 const Multinomial &alternate_distribution) {
    double sum = Sum(*p_);
    if(sum > 0) {
      la::Scale(((double)1) / sum, p_);
    }
    else {
      p_ -> CopyValues(*(alternate_distribution.p_));
    }

    UpdateLogP();
  }
  
  void PrintDebug(const char *name = "", FILE *stream = stderr) const {
    fprintf(stream, name);
    p_ -> PrintDebug("p");
    logp_ -> PrintDebug("logp");
  }

  ~Multinomial() {
    Destruct();
  }

  void Destruct() {
    delete p_;
    delete logp_;
  }

};
    
#endif /* MULTINOMIAL_H */

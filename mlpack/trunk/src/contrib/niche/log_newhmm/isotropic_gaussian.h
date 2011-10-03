#ifndef ISOTROPIC_GAUSSIAN_H
#define ISOTROPIC_GAUSSIAN_H

#include "fastlib/fastlib.h"

class IsotropicGaussian {

 private:
  double norm_constant_;
  double min_variance_;

 public:

  int n_dims_;

  Vector* mu_;
  double sigma_;

  void Init(int n_dims_in) {
    Init(n_dims_in, 0);
  }
  
  void Init(int n_dims_in, double min_variance_in) {
    Init(n_dims_in, min_variance_in, 1);
  }

  void Init(int n_dims_in, double min_variance_in, int n_components_in) {
    n_dims_ = n_dims_in;
    min_variance_ = min_variance_in;

    mu_ = new Vector();
    mu_ -> Init(n_dims_);
  }
  
  void Init(const Vector &mu_in,
	    const double sigma_in) {
    mu_ = new Vector();
    mu_ -> Copy(mu_in);
    
    sigma_ = sigma_in;

    n_dims_ = mu_ -> length();
  }
  
  void CopyValues(const IsotropicGaussian &other) {
    mu_ -> CopyValues(other.mu());
    sigma_ = other.sigma();
  }
  
  int n_dims() const {
    return n_dims_;
  }

  const Vector mu() const {
    return *mu_;
  }

  double sigma() const {
    return sigma_;
  }
  
  void SetMu(const Vector &mu_in) {
    mu_ -> CopyValues(mu_in);
  }

  void SetSigma(double sigma_in) {
    sigma_ = sigma_in;
  }
  
  void RandomlyInitialize() {
    for(int i = 0; i < n_dims_; i++) {
      (*mu_)[i] = drand48();
    }
    sigma_ = 1;
    ComputeNormConstant();
  }

  void ComputeNormConstant() {
    norm_constant_ = 
      1 / pow(2 * M_PI * sigma_, ((double)n_dims_) / ((double)2));
  }

  template<typename T>
  double PkthComponent(const GenVector<T> &x, int component_num) {
    return Pdf(x);
  }
  
  template<typename T>
  double Pdf(const GenVector<T> &x) {
    return exp(-0.5 * la::DistanceSqEuclidean(x, *mu_) / sigma_)
      * norm_constant_;
  }

  void PrintDebug(const char *name = "", FILE *stream = stderr) const {
    fprintf(stream, name);
    mu_ -> PrintDebug("mu");
    printf("sigma = %f\n", sigma_);
  }
    
  
  ~IsotropicGaussian() {
    Destruct();
  }

  void Destruct() {
    delete mu_;
  }


};
    
#endif /* ISOTROPIC_GAUSSIAN_H */

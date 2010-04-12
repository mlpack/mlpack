#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

class Distribution {
 public:


  int n_dims_;

  Vector mu_;
  Matrix sigma_;


  void Init(int n_dims_in) {
    n_dims_ = n_dims_in;
    mu_.Init(n_dims_);
    sigma_.Init(n_dims_, n_dims_);
  }


  void RandomlyInitialize() {
    sigma_.SetZero();
    for(int i = 0; i < n_dims_; i++) {
      mu_[i] = drand48();
      
      // spherical covariance
      sigma_.set(i, i, 1);
    }
  }

  int n_dims() {
    return n_dims_;
  }
  
  Vector mu() {
    return mu_;
  }

  Matrix sigma() {
    return sigma_;
  }

  void SetMu(Vector mu_in) {
    mu_.CopyValues(mu_in);
  }

  void SetSigma(Matrix sigma_in) {
    sigma_.CopyValues(sigma_in);
  }

  void PrintDebug(char *name) {
    printf("----- DISTRIBUTION %s -----\n", name);
    mu_.PrintDebug("mu");
    sigma_.PrintDebug("sigma");
  }
};


#endif /* DISTRIBUTION_H */

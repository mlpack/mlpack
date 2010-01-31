#ifndef MIXTURE_H
#define MIXTURE_H

#include "utils.h"

template <typename TDistribution>
class Mixture {

 private:

  int n_components_;
  int n_dims_;

  Vector weights_;



 public:
  TDistribution* components_;

  void Init(int n_dims_in, double min_variance_in, int n_components_in) {
    n_dims_ = n_dims_in;
    n_components_ = n_components_in;

    weights_.Init(n_components_);
    components_ =
      (TDistribution*) malloc(n_components_ * sizeof(TDistribution));
    for(int i = 0; i < n_components_; i++) {
      components_[i].Init(n_dims_, min_variance_in);
    }
  }

  void RandomlyInitialize() {
    weights_.SetAll(((double)1) / ((double)n_components_));
    for(int k = 0; k < n_components_; k++) {
      components_[k].RandomlyInitialize();
    }
  }

  template<typename T>
  double PkthComponent(const GenVector<T> &xt, int component_num) {
    return weights_[component_num] * components_[component_num].Pdf(xt);
  }

  template<typename T>
  double LogPkthComponent(const GenVector<T> &xt, int component_num) {
    return weights_[component_num] + components_[component_num].LogPdf(xt);
  }

  double Pdf(const Vector &xt) {
    double sum = 0;
    for(int k = 0; k < n_components_; k++) {
      sum += PkthComponent(xt, k);
    }
    return sum;
  }

  double LogPdf(const Vector &xt) {
    double sum = LogPkthComponent(xt, 0);
    for(int k = 1; k < n_components_; k++) {
      sum = LogSumExp(sum, LogPkthComponent(xt, k));
    }
    return sum;
  }

  void SetZero() {
    for(int k = 0; k < n_components_; k++) {
      components_[k].SetZero();
    }
    weights_.SetZero();
  }
  
  void Accumulate(double weight, const Vector &example,
		  int component_num) {
    components_[component_num].Accumulate(weight, example, 0);
    weights_[component_num] += weight;
  }

  void Normalize(double one_over_normalization_factor) {
    for(int k = 0; k < n_components_; k++) {
      components_[k].Normalize(weights_[k]);
    }
    
    // I posit that one_over_normalization_factor = sum(weights_)
    la::Scale(((double)1) / one_over_normalization_factor, &weights_);
  }
  
  void Normalize(double one_over_normalization_factor,
		 const Mixture &alternate_distribution) {
    for(int k = 0; k < n_components_; k++) {
      components_[k].Normalize(weights_[k],
			       alternate_distribution.components_[k]);
    }
    
    // I posit that one_over_normalization_factor = sum(weights_)
/*     printf("one_over_normalization_factor = %f\n", one_over_normalization_factor); */
/*     printf("weights = ["); */
/*     for(int k = 0; k < n_components_; k++) { */
/*       printf("%f ", weights_[k]); */
/*     } */
/*     printf("]\n"); */
    if(one_over_normalization_factor > 0) {
      la::Scale(((double)1) / one_over_normalization_factor, &weights_);
    }
    else {
      weights_.CopyValues(alternate_distribution.weights_);
    }
  }

  void PrintDebug(const char *name = "", FILE *stream = stderr) const {
    fprintf(stream, name);
    weights_.PrintDebug("weights");
    char string[100];
    for(int k = 0; k < n_components_; k++) {
      sprintf(string, "component %d\n", k + 1);
      components_[k].PrintDebug(string);
    }
  }
  
  ~Mixture() {
    Destruct();
  }

  void Destruct() {
    for(int i = 0; i < n_components_; i++) {
      components_[i].Destruct();
    }
    free(components_);
    //weights_.Destruct();
  }

};


#endif /* MIXTURE_H */
  

  
    

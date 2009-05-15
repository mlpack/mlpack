#ifndef MIXTURE_H
#define MIXTURE_H

template <typename TDistribution>
class Mixture {

 private:

  int n_components_;
  int n_dims_;

  Vector weights_;
  TDistribution* components_;



 public:

  void Init(int n_dims_in, int n_components_in) {
    n_dims_ = n_dims_in;
    n_components_ = n_components_in;

    
    weights_.Init(n_components_);
    components_ =
      (TDistribution*) malloc(n_components_ * sizeof(TDistribution));
    for(int i = 0; i < n_components_; i++) {
      components_[i].Init(n_dims_);
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

  double Pdf(const Vector &xt) {
    double sum = 0;
    for(int k = 0; k < n_components_; k++) {
      sum += PkthComponent(xt, k);
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
  
  void Normalize(double normalization_factor) {
    for(int k = 0; k < n_components_; k++) {
      double one_over_weights_k = ((double)1) / weights_[k];
      components_[k].Normalize(one_over_weights_k);
    }
    
    // I posit that normalization_factor = sum(weights_)
    la::Scale(normalization_factor, &weights_);
  }

  void PrintDebug(const char *name = "", FILE *stream = stderr) const {
    fprintf(stream, name);
    weights_.PrintDebug("weights");
    char string[100];
    for(int k = 0; k < n_components_; k++) {
      sprintf(string, "component %d", k + 1);
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
  }

};


#endif /* MIXTURE_H */
  

  
    

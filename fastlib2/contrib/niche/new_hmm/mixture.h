#ifndef MIXTURE_H
#define MIXTURE_H

template <typename TDistribution>
class Mixture {

 private:

  int n_components_;
  int n_dims_;

  Vector weights;
  TDistribution* components_;



 public:

  void Init(int n_components_in, int n_dims_in) {
    n_components_ = n_components_in;
    n_dims_ = n_dims_in;
    
    weights.Init(n_components_);
    components_ =
      (TDistribution*) malloc(n_components_ * sizeof(TDistribution));
    for(int i = 0; i < n_components_; i++) {
      components_[i].Init(n_dims_);
    }
  }

  void Destruct() {
    for(int i = 0; i < n_components_; i++) {
      components_[i].Destruct();
    }
    free(components_);
  }

};


#endif /* MIXTURE_H */
  

  
    

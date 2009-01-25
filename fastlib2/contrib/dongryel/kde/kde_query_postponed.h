#ifndef INSIDE_KDE_PROBLEM_H
#error "This is not a public header file!"
#endif

class MultiTreeQueryPostponed {
    
 public:
  
  double sum_l;
  double sum_e;
  double n_pruned;
  double used_error;
  double probabilistic_used_error;
  
 public:

  template<typename TDelta>
  void ApplyDelta(const TDelta &delta_in) {
    sum_l += delta_in.kde_approximation.sum_l;
    sum_e += delta_in.kde_approximation.sum_e;
    n_pruned += delta_in.kde_approximation.n_pruned;
    used_error += delta_in.kde_approximation.used_error;
    probabilistic_used_error =
      sqrt(math::Sqr(probabilistic_used_error) +
	   math::Sqr(delta_in.kde_approximation.probabilistic_used_error));
  }
    
  template<typename TQueryPostponed>
  void ApplyPostponed(const TQueryPostponed &postponed_in) {
    sum_l += postponed_in.sum_l;
    sum_e += postponed_in.sum_e;
    n_pruned += postponed_in.n_pruned;
    used_error += postponed_in.used_error;
    probabilistic_used_error =
      sqrt(math::Sqr(probabilistic_used_error) +
	   math::Sqr(postponed_in.probabilistic_used_error));
  }
    
  void SetZero() {
    sum_l = 0;
    sum_e = 0;
    n_pruned = 0;
    used_error = 0;
    probabilistic_used_error = 0;
  }
    
};

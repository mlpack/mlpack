#ifndef MMK_H
#define MMK_H

template <typename T>
class MeanMapKernel {

 private:
  double lambda_;
  int n_T_;
  
 public:
  void Init(double lambda_in) {
    Init(lambda_in, 1);
  }
  
  void Init(double lambda_in, double n_T_in) {
    lambda_ = lambda_in;
    n_T_ = n_T_in;
  }
  
  double Compute(T a, T b);
};

template <typename T> double MeanMapKernel<T>::Compute(T a, T b) {
  return T::Compute(a, b, lambda_, n_T_);
}

#endif /* MMK_H */

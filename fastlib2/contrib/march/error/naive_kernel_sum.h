#ifndef NAIVE_KERNEL_SUM_H
#define NAIVE_KERNEL_SUM_H


#include "fastlib/fastlib.h"


class NaiveKernelSum {

 private:

  Vector results_;
  
  Matrix centers_;
  
  double bandwidth_;
  
  index_t num_points_;
  
  index_t dimension_;
  
  struct datanode* module_;
  
  double ComputeGaussian_(index_t i, index_t j) {
  
    Vector i_vec;
    centers_.MakeColumnVector(i, &i_vec);
    
    Vector j_vec;
    centers_.MakeColumnVector(j, &j_vec);
    
    double dist_sq = la::DistanceSqEuclidean(i_vec, j_vec);
    
    return (exp(-bandwidth_ * dist_sq));
  
  } // ComputeGaussian_()
  
 public:

  NaiveKernelSum() {}
  
  ~NaiveKernelSum() {}
  
  void Init(struct datanode* mod, const Matrix& cent, double band) {
  
    centers_.Copy(cent);
    
    bandwidth_ = band;
    DEBUG_ASSERT(bandwidth_ > 0.0);
    
    num_points_ = centers_.n_cols();
    
    dimension_ = centers_.n_rows();
    
    results_.Init(num_points_);
    results_.SetZero();
    
    module_ = mod;
  
  } // Init()
  
  void ComputeTotalSum(Vector* return_results) {
  
    fx_timer_start(module_, "naive_time");
  
    for (index_t i = 0; i < num_points_; i++) {
    
      double this_result = results_[i];
  
      for (index_t j = i; j < num_points_; j++) { // for symmetry
      
        double this_kernel = ComputeGaussian_(i, j);
        this_result = this_result + this_kernel;
        
        results_[j] = results_[j] + this_kernel; // take advantage of symmetry
      
      } // j
      
      results_[i] = this_result;
      //printf("results[%d] = %g\n", i, this_result);
    
    } // i
    
    fx_timer_stop(module_, "naive_time");
    
    return_results->Copy(results_);
  
  } // ComputeTotalSum()
  
  void NaiveComputation(const char* dataset, const char* output, 
                        Vector* naive_results) {
  
    ComputeTotalSum(naive_results);
        
    Matrix out_mat;
    out_mat.AliasColVector(*naive_results);
    data::Save(output, out_mat);
    
  
  } // NaiveComputation()

}; // NaiveKernelSum


#endif

#ifndef HYBRID_ERROR_ANALYSIS_H
#define HYBRID_ERROR_ANALYSIS_H

#include "fastlib/fastlib.h"

class ErrorAnalysis {

private:

  Vector kernel_vec_;
  Vector naive_vec_;
  
  struct datanode* kernel_mod_;
  struct datanode* naive_mod_;
  
  index_t num_points_;
  
  void TotalError_() {
  
    double kernel_error = 0.0;
        
    double kernel_rel = 0.0;
    
    double kernel_error_max = 0.0;

    double kernel_error_max_rel = 0.0;
    
    double average_abs_error;
    double average_rel_error;
    
    /*
    printf("kernel vec:\n");
    ot::Print(kernel_vec_);
    printf("\n\n naive_vec:\n");
    ot::Print(naive_vec_);
    printf("\n");
    */
    
    for (index_t i = 0; i < num_points_; i++) { 
    
      double naive_val = naive_vec_[i];
      
      double this_kernel = fabs(naive_val - kernel_vec_[i]);
      
      if (this_kernel > kernel_error_max) {
        kernel_error_max = this_kernel;
      }
      
      kernel_error = kernel_error + this_kernel;
      
      double this_rel = this_kernel/naive_val;
      
      if (this_rel > kernel_error_max_rel) {
        kernel_error_max_rel = this_rel;
        //printf("i=%d, naive=%g, kernel=%g\n", i, naive_val, kernel_vec_[i]);
      }
      
      kernel_rel = kernel_rel + this_rel;
    
    } // i
    
    average_abs_error = kernel_error/num_points_;
    average_rel_error = kernel_rel/num_points_;
    
    fx_format_result(kernel_mod_, "total_absolute_error", "%g", kernel_error);
    
    fx_format_result(kernel_mod_, "total_relative_error", "%g", kernel_rel);
    
    fx_format_result(kernel_mod_, "max_absolute_error", "%g", kernel_error_max);  
  
    fx_format_result(kernel_mod_, "max_relative_error", "%g", 
                     kernel_error_max_rel);
                     
    fx_format_result(kernel_mod_, "ave_abs_error", "%g", average_abs_error);
    
    fx_format_result(kernel_mod_, "ave_rel_error", "%g", average_rel_error);
                     
                     
  
  } // TotalAbsoluteError_()
  
  
 public:

  ErrorAnalysis() {}
  
  ~ErrorAnalysis() {}
  
  void Init(const Vector& kern, const Vector& naive, struct datanode* kernel_m,
            struct datanode* naive_m) {
  
    
    kernel_vec_.Copy(kern);
    naive_vec_.Copy(naive);
    
    num_points_ = kernel_vec_.length();
    
    kernel_mod_ = kernel_m;
    naive_mod_ = naive_m;
  
  } // Init()
  
  
  void ComputeResults() {
  
    TotalError_();
  
  } // ComputeResults()
  


}; // class ErrorAnalysis


#endif
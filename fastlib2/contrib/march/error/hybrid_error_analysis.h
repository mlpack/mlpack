#ifndef HYBRID_ERROR_ANALYSIS_H
#define HYBRID_ERROR_ANALYSIS_H

#include "fastlib/fastlib.h"

class ErrorAnalysis {

private:

  Vector abs_vec;
  Vector rel_vec;
  Vector exp_vec;
  Vector gauss_vec;
  Vector hybrid_vec;
  Vector naive_vec;
  
  struct datanode* abs_mod_;
  struct datanode* rel_mod_;
  struct datanode* exp_mod_;
  struct datanode* gauss_mod_;
  struct datanode* hybrid_mod_;
  struct datanode* naive_mod_;
  
  index_t num_points_;
  
  void TotalError_() {
  
    double abs_error = 0.0;
    double rel_error = 0.0;
    double exp_error = 0.0;
    double gauss_error = 0.0;
    double hybrid_error = 0.0;
    
    double abs_rel = 0.0;
    double rel_rel = 0.0;
    double exp_rel = 0.0;
    double gauss_rel = 0.0;
    double hybrid_rel = 0.0;
    
    double abs_error_max = 0.0;
    double rel_error_max = 0.0;
    double exp_error_max = 0.0;
    double gauss_error_max = 0.0;
    double hybrid_error_max = 0.0;

    double abs_error_max_rel = 0.0;
    double rel_error_max_rel = 0.0;
    double exp_error_max_rel = 0.0;
    double gauss_error_max_rel = 0.0;
    double hybrid_error_max_rel = 0.0;
    
    
    for (index_t i = 0; i < num_points_; i++) { 
    
      double naive_val = naive_vec[i];
      
      double this_abs = fabs(naive_val - abs_vec[i]);
      double this_rel = fabs(naive_val - rel_vec[i]);
      double this_exp = fabs(naive_val - exp_vec[i]);
      double this_gauss = fabs(naive_val - gauss_vec[i]);
      double this_hybrid = fabs(naive_val - hybrid_vec[i]);
      
      if (this_abs > abs_error_max) {
        abs_error_max = this_abs;
      }
      if (this_rel > rel_error_max) {
        rel_error_max = this_rel;
      }
      if (this_exp > exp_error_max) {
        exp_error_max = this_exp;
      }
      if (this_gauss > gauss_error_max) {
        gauss_error_max = this_gauss;
      }
      if (this_hybrid > hybrid_error_max) {
        hybrid_error_max = this_hybrid;
      }
      
      
      this_abs = this_abs/naive_val;
      this_rel = this_rel/naive_val;
      this_exp = this_exp/naive_val;
      this_gauss = this_gauss/naive_val;
      this_hybrid = this_hybrid/naive_val;
      
      if (this_abs > abs_error_max_rel) {
        abs_error_max_rel = this_abs;
      }
      if (this_rel > rel_error_max_rel) {
        rel_error_max_rel = this_rel;
      }
      if (this_exp > exp_error_max_rel) {
        exp_error_max_rel = this_exp;
      }
      if (this_gauss > gauss_error_max_rel) {
        gauss_error_max_rel = this_gauss;
      }
      if (this_hybrid > hybrid_error_max_rel) {
        hybrid_error_max_rel = this_hybrid;
      }
      
      abs_error = abs_error + this_abs;
      rel_error = rel_error + this_rel;
      exp_error = exp_error + this_exp;
      gauss_error = gauss_error + this_gauss;
      hybrid_error = hybrid_error + this_hybrid;
      
      abs_rel = abs_rel + (this_abs/naive_val);
      rel_rel = rel_rel + (this_rel/naive_val);
      exp_rel = exp_rel + (this_exp/naive_val);
      gauss_rel = gauss_rel + (this_gauss/naive_val);
      hybrid_rel = hybrid_rel + (this_gauss/naive_val);
    
    } // i
    
    fx_format_result(abs_mod_, "total_absolute_error", "%g", abs_error);
    fx_format_result(rel_mod_, "total_absolute_error", "%g", rel_error);
    fx_format_result(exp_mod_, "total_absolute_error", "%g", exp_error);
    fx_format_result(gauss_mod_, "total_absolute_error", "%g", gauss_error);
    fx_format_result(hybrid_mod_, "total_absolute_error", "%g", hybrid_error);
    
    fx_format_result(abs_mod_, "total_relative_error", "%g", abs_rel);
    fx_format_result(rel_mod_, "total_relative_error", "%g", rel_rel);
    fx_format_result(exp_mod_, "total_relative_error", "%g", exp_rel);
    fx_format_result(gauss_mod_, "total_relative_error", "%g", gauss_rel);
    fx_format_result(hybrid_mod_, "total_relative_error", "%g", hybrid_rel);
    
    fx_format_result(abs_mod_, "max_absolute_error", "%g", abs_error_max);
    fx_format_result(rel_mod_, "max_absolute_error", "%g", rel_error_max);
    fx_format_result(exp_mod_, "max_absolute_error", "%g", exp_error_max);
    fx_format_result(gauss_mod_, "max_absolute_error", "%g", gauss_error_max);
    fx_format_result(hybrid_mod_, "max_absolute_error", "%g", hybrid_error_max);  
  
    fx_format_result(abs_mod_, "max_relative_error", "%g", abs_error_max_rel);
    fx_format_result(rel_mod_, "max_relative_error", "%g", rel_error_max_rel);
    fx_format_result(exp_mod_, "max_relative_error", "%g", exp_error_max_rel);
    fx_format_result(gauss_mod_, "max_relative_error", "%g", 
                     gauss_error_max_rel);
    fx_format_result(hybrid_mod_, "max_relative_error", "%g", 
                     hybrid_error_max_rel);
  
  } // TotalAbsoluteError_()
  
  
 public:

  ErrorAnalysis() {}
  
  ~ErrorAnalysis() {}
  
  void Init(const Vector& abs, const Vector& rel, const Vector& exp, 
            const Vector& gauss, const Vector& hybrid, const Vector& naive, 
            struct datanode* abs_m, struct datanode* rel_m, 
            struct datanode* exp_m, struct datanode* gauss_m, 
            struct datanode* hybrid_m, struct datanode* naive_m) {
  
    abs_vec.Copy(abs);
    rel_vec.Copy(rel);
    exp_vec.Copy(exp);
    gauss_vec.Copy(gauss);
    hybrid_vec.Copy(hybrid);
    naive_vec.Copy(naive);
    
    num_points_ = abs_vec.length();
    
    abs_mod_ = abs_m;
    rel_mod_ = rel_m;
    exp_mod_ = exp_m;
    gauss_mod_ = gauss_m;
    hybrid_mod_ = hybrid_m;
    naive_mod_ = naive_m;
  
  } // Init()
  
  
  void ComputeResults() {
  
    TotalError_();
  
  } // ComputeResults()
  


}; // class ErrorAnalysis


#endif
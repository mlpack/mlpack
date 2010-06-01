#ifndef OCAS_H_
#define OCAS_H_
#include "ocas_smo.h"



class OCAS{

private:
  
  // The dataset that will be used for optimization. This will have
  // points outside the range of the query point ALSO.
  Matrix train_data_appended_;

  // The query point where we want to learn the local SVM.
  Vector query_point_appended_;
  
  // The bandwidth of the smoothing kernel.

  double smoothing_kernel_bandwidth_;

  // The regularization constant

  double lambda_reg_const_;

  // The subgradients and the intercepts

  ArrayList<Vector> subgradients_mat_;
  ArrayList <double> intercepts_vec_;
 
  
  // The indices that are in the bandwidth range.

  ArrayList <index_t> indices_in_range_;

  // The kernel value of the points which are in the range.

  ArrayList <double> smoothing_kernel_values_in_range_;


  // This is the dimension of the appended dataset. Hence it will be 1 more 
  // than the actual dimensions.

  index_t num_dims_appended_;

  // The number of train points

  index_t num_train_points_;

  // The number of points in the range 

  index_t num_points_in_range_;

  // The train labels

  Vector train_labels_;
  

  // OCAS specific quantities

  Vector w_best_at_t_;
  Vector w_smo_at_t_;
  Vector w_c_at_t_;
  

  // The constant used in the algorithm

  double lambda_ocas_const_;

  
  // Number of subgradients available

  index_t num_subgradients_available_;
  

 
 private:

  double CalculateNonDifferentiablePartOfObjective_(Vector &);
  double CalculateApproximatedObjectiveValueAtAPoint_(Vector &);
  double CalculateObjectiveFunctionValueAtAPoint_(Vector &);
  void GetWCAtT_();
  void GetWBAtT_(double);
  void GetSubGradientAndInterceptAtNewPoint_();
  double DoLineSearch_();
  void SMOMainRoutine_();
  void PrintSubgradientsAndIntercepts_();
 public:
    
  void Optimize();  
  void Init(double *, Matrix &,Vector&,
	    ArrayList <index_t> &, 
	    ArrayList <double> &, double bw, 
	    double lambda_reg_const);
  

  // The getter function
  void get_optimal_vector(Vector &);
};

#endif

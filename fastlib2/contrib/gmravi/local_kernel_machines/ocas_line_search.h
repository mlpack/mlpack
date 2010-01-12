#include "fastlib/fastlib.h"
#ifndef OCAS_LINE_SEARCH_H_
#define OCAS_LINE_SEARCH_H_
class Interval{

 public:

  double up;
  double low;

  // some useful functions

  bool CheckIfIntervalHasZero(){

    double low=this->low;
    double up=this->up;
    if(low<0 && up>0) return true;
    else{
      
      if(fabs(low*up)<SMALL*SMALL&&low*up<0.0){
	printf("low=%f,up=%f...\n",low,up);
	return true;
      }
      return false;
    }
  }

  bool CheckIfIntervalIsPositive(){
    if(this->low>0&& this->up>0) return true;
    return false;

  }

  bool CheckIfIntervalIsNegative(){
    if(this->low<0&& this->up<0) return true;
    return false;

  }
};
class OCASLineSearch{

 public:

  
  // The dataset that will be used for optimization. This will have
  // points outside the range of the query point ALSO.
  Matrix train_data_appended_;

  // The train labels of all points
  Vector train_labels_;

  
  // The query point where we want to learn the local SVM.
  Vector query_point_appended_;
  
  
  // The regularization constant
  
  double lambda_reg_const_;


  // The indices that are in the bandwidth range
  
  ArrayList <index_t> indices_in_range_;


  // The kernel value of the points which are in the range.

  ArrayList <double> smoothing_kernel_values_in_range_;


  // This is the dimension of the appended dataset. Hence it will be 1 more 
  // than the actual dimensions.

  index_t num_dims_appended_;
  
  // The number of points in the range 
  
  index_t num_points_in_range_;

  // The total number of points
  index_t num_train_points_;


 
  // The thresholds

  Vector thresholds_vec_;

  // Initialize the derivatives of the line search objective

  ArrayList <Interval> derivative_line_search_objective_;

  //The vector $w_1$

  Vector w_1_vec_;

  // The vector $w_2$

  Vector w_2_vec_;

  // The optimal k. i.e we are trying to minimize the function
  // F[(1-k)w_1+kw_2] over k.

  double k_star_;

  // A constant that is used heavily while calculating derivatives

  double lambda_w_2_minus_w_1_sqd_;

  // Vector C_i and B_i. These will be used in gradient calculations as well as
  // threshold calculations

  Vector C_i_vec_;

  Vector B_i_vec_;

  // The getter function
  double get_optimal_k();

  
  
  void CalculateThresholdsForPointsInRange_();
  void SortThresholds_();
  void Swap_(int *a,int *b);
  void Swap_(double *a,double *b);
  void CalculateDerivativesAtEachThreshold(int);
  void CalculateDerivatives_();  
  double CalculateOptimalK_();
  double ComputeGradientBeyondAndOptimalK_(int);
  double ComputeGradientBeforeAndOptimalK_(int index);
  double ComputeGradientInIntervalAndOptimalK_(int index);
  void  CalculateGradientAtAPoint(double k,Interval &derivative_interval);
 public:
  void Init(Matrix &train_data_appended,Vector &train_labels, 
	    Vector &query_point_appended, double lambda_reg_const, 
	    ArrayList<index_t> &indices_in_range,
	    ArrayList <double> &smoothing_kernel_values_in_range, 
	    int num_points_in_range, int num_points, Vector &w_1,Vector &w_2);
  
  void PerformLineSearch();

    
  
};

#endif

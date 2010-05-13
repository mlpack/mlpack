#include "fastlib/fastlib.h"
#include "ocas.h"
#include "ocas_line_search.h"


/** A getter utility that returns the value of optimal k
 */

double OCASLineSearch::get_optimal_k(){
  
  return k_star_;
}
void OCASLineSearch::Swap_(int *a,int *b){

  int temp=*a;
  *a=*b;
  *b=temp;
}


void OCASLineSearch::Swap_(double *a,double *b){

  double temp=*a;
  *a=*b;
  *b=temp;
}

void  OCASLineSearch::CalculateThresholdsForPointsInRange_(){


  // C_i=K_i(1-y_i<w_1,x_i>)/n
  // B_i=-K_iy_i <w_2-w_1,x_i>/n
  // threshold= -C_i/B_i=(1-y_i<w_1,x_i>)/y_i<w_2-w_1,x_i>
  
  //  w_1_vec_.PrintDebug();
  // w_2_vec_.PrintDebug();

  for(int i=0;i<num_points_in_range_;i++){

    int original_index=indices_in_range_[i];
    Vector w_2_minus_w_1;
    la::SubInit(w_1_vec_, w_2_vec_, &w_2_minus_w_1);
    
    Vector x_i;

    train_data_appended_.MakeColumnVector(original_index,&x_i); 

    double w_2_minus_w_1_dot_x_i=la::Dot (w_2_minus_w_1,x_i);
    double w_1_dot_x_i=la::Dot(w_1_vec_,x_i);
    
    double y_lab=train_labels_[original_index];

    // Set C_i_vec and B_i_vec

    C_i_vec_[i]=
      smoothing_kernel_values_in_range_[i]*(1-y_lab*w_1_dot_x_i)/
      num_train_points_;
    
    B_i_vec_[i]=-y_lab*w_2_minus_w_1_dot_x_i*
      smoothing_kernel_values_in_range_[i]/num_train_points_;
     
    
    if(fabs(B_i_vec_[i])<SMALL){
      
      thresholds_vec_[i]=0;
    }
    else{
      thresholds_vec_[i]=-1*C_i_vec_[i]/B_i_vec_[i];
    }
  }
}

// This function sorts the thresholds and at the same time also
// permutes the indices which are in range

// Implementing simple bubble sort
void OCASLineSearch::SortThresholds_(){

  
  for(int i=0;i<num_points_in_range_;i++){

    for(int j=i+1;j<num_points_in_range_;j++){
            
      if(thresholds_vec_[i]>thresholds_vec_[j]){
	
	
	Swap_(&thresholds_vec_[i],&thresholds_vec_[j]);
	
	
	//At the same time swap indices also and C_i and B_i
	  
	Swap_(&indices_in_range_[i],&indices_in_range_[j]);
	Swap_(&C_i_vec_[i],&C_i_vec_[j]);
	Swap_(&B_i_vec_[i],&B_i_vec_[j]);
	Swap_(&smoothing_kernel_values_in_range_[i],
	  &smoothing_kernel_values_in_range_[j]);
      }
    }
  }
  
  for(int i=0;i<num_points_in_range_-1;i++){
    
    if(thresholds_vec_[i]>thresholds_vec_[i+1]){
      
      printf("Sorting was incorrect....\n");
      printf("thresholds_vec_[%d]=%f,thresholds_vec_[%d]=%f...\n",i,thresholds_vec_[i],i,thresholds_vec_[i+1]);
      exit(0);
    }
  }
}


// Remember this index is w.r.t the vector indices_in_range_(after
// having sorted)
void OCASLineSearch::CalculateDerivativesAtEachThreshold(int index){

  // The derivative has 2 parts. The first part has 2 contributions.
  
  Vector w_2_minus_w_1;
  la::SubInit(w_1_vec_, w_2_vec_, &w_2_minus_w_1);
   
  double B_0=lambda_reg_const_*la::Dot(w_1_vec_,w_2_minus_w_1);

  double delta_g_0_k=
    lambda_w_2_minus_w_1_sqd_*thresholds_vec_[index]+B_0;
  
  
  double delta_g_i_k=0.0;
  for(int i=0;i<num_points_in_range_;i++){
        
    if(i<index){

      // The contribution is max(0,B_i)
      
      delta_g_i_k+=max(0.0,B_i_vec_[i]);
    }
    if(i>index){

      // The contribution is min(0,B_i)
      delta_g_i_k+=min(0.0,B_i_vec_[i]); 
    }    
  }
  // To this add the contribution of point "index"
  derivative_line_search_objective_[index].low=
    delta_g_0_k+delta_g_i_k+min(0.0,B_i_vec_[index]);

  derivative_line_search_objective_[index].up=
    delta_g_0_k+delta_g_i_k+max(0.0,B_i_vec_[index]);
}

// This function simply calculates the derivatives at each and every
// threshold

void OCASLineSearch::CalculateDerivatives_(){
  
  for(int i=0;i<num_points_in_range_;i++){

    CalculateDerivativesAtEachThreshold(i);    
  }

  // This is just a test to make sure that the derivatives are increasing

  for(int l=0;l>num_points_in_range_-1;l++){
    
    if(derivative_line_search_objective_[l].low>
       derivative_line_search_objective_[l].up+SMALL){
      
      printf("derivative calculation mistake..\n");
      printf("derivative_line_search_objective_[%d].low=%f...\n",
	     l,derivative_line_search_objective_[l].low);
      printf("derivative_line_search_objective_[%d].up=%f...\n",
	     l,derivative_line_search_objective_[l].up);
      
    }
    
    double prev_up=derivative_line_search_objective_[l].up;
    double next_low=derivative_line_search_objective_[l+1].low;

    if(prev_up>next_low+SMALL){
      
      printf("derivative calculation mistake...\n");
      
      printf("prev_up=%f...\n",prev_up);
      printf("next_low=%f..\n",next_low);
    }
  }
}

/** Computing gradient beyond the given index but before the next index
 */



double OCASLineSearch::ComputeGradientInIntervalAndOptimalK_(int index){

  // The gradient in this region is a linear function of k. We are
  // required to solve this linear equation

  double linear_coeff=lambda_w_2_minus_w_1_sqd_;

  // We shall reuse the derivative at the threshold provided by index
  // to calculate the linear part. Alternatively this can be
  // calculated from scratch

  double constant=derivative_line_search_objective_[index].up-
    (lambda_w_2_minus_w_1_sqd_*thresholds_vec_[index]);

  // The optimal k is  simply max(0,-constant/linear_coeff);
  
  double k_opt=max(0.0,-constant/linear_coeff);
  printf("k_opt is %f..\n",k_opt);
  printf("Investigating index=%d..\n",index);
  printf("the next threshold is %f..\n",thresholds_vec_[index+1]);

  if(k_opt-thresholds_vec_[index+1]>=SMALL){
    printf("There is a mistake in line search calculations...\n");

    for(int i=0;i<num_points_in_range_;i++){
      printf("threshold=%f,derivative=[%f,%f]...\n",
	     thresholds_vec_[i],derivative_line_search_objective_[i].low,
	     derivative_line_search_objective_[i].up);
    }

    exit(0);
  }
  return k_opt;
}


/** This function has been called because the optimal k seems to lie
    beyond the threshold given by the argument index

*/

double OCASLineSearch::ComputeGradientBeyondAndOptimalK_(int index){

  // The gradient in this region is a linear function of k. We are
  // required to solve this linear equation

  double linear_coeff=lambda_w_2_minus_w_1_sqd_;

  // We shall reuse the derivative at the threshold provided by index
  // to calculate the linear part. Alternatively this can be
  // calculated from scratch

  double constant=derivative_line_search_objective_[index].up-
    (lambda_w_2_minus_w_1_sqd_*thresholds_vec_[index]);

  // The optimal k is  simply max(0,-constant/linear_coeff);
  
  double k_opt=max(0.0,-constant/linear_coeff);
  return k_opt;
}

double OCASLineSearch::ComputeGradientBeforeAndOptimalK_(int index){

  double linear_coeff=lambda_w_2_minus_w_1_sqd_;
  
  // We shall reuse the derivative at the threshold provided by index
  // to calculate the linear part. Alternatively this can be
  // calculated from scratch
  
  double constant=derivative_line_search_objective_[index].low-
    (lambda_w_2_minus_w_1_sqd_*thresholds_vec_[index]);

  // The optimal k is  simply max(0,-constant/linear_coeff);

  double k_opt=max(0.0,-constant/linear_coeff);
  return k_opt;
}

void OCASLineSearch::CalculateGradientAtAPoint(double k,Interval &derivative_interval){
  
  // The derivative has 2 parts. The first part has 2 contributions.
  

  double kA0= k*lambda_w_2_minus_w_1_sqd_;

  Vector w_2_minus_w_1;
  la::SubInit(w_1_vec_, w_2_vec_, &w_2_minus_w_1);
   
  double B_0=lambda_reg_const_*la::Dot(w_1_vec_,w_2_minus_w_1);
  derivative_interval.up=B_0+kA0;
  derivative_interval.low=B_0+kA0;

  for(int i=0;i<num_points_in_range_;i++){
    
    //    int original_index=indices_in_range_[index];
    
    double k_i=thresholds_vec_[i];

    if(fabs(k_i-k)<SMALL){
      
      // The contribution is [0,B_i]
      // The contribution is max(0,B_i)
      derivative_interval.low+=min(0.0,B_i_vec_[i]); 
      derivative_interval.up+=max(0.0,B_i_vec_[i]); 
    }
    else{
      
      if(k_i<k){
	
	// Contribution is max(0,B_i)
	derivative_interval.low+=max(0.0,B_i_vec_[i]);
	derivative_interval.up+=max(0.0,B_i_vec_[i]);
	
      }
      else{
	
	// Contribution is min(0,B_i)
	derivative_interval.low+=min(0.0,B_i_vec_[i]);
	derivative_interval.up+=min(0.0,B_i_vec_[i]);

      } 
    }    
  }
}

double OCASLineSearch::CalculateOptimalK_(){
   
  
  // The first thing to do is calculate the derivative at k=0

  Interval derivative_interval;

  CalculateGradientAtAPoint(0.0,derivative_interval);

  if(derivative_interval.up>0||fabs(derivative_interval.up)<SMALL){
    
    // In this case the optimal solution is k_star=0;
    return 0.0;
    
  }
  else{// The gradient at 0 is entirely negative 
    // Check out the gradient at the right most threshold
    
    Interval derivative_interval=
      derivative_line_search_objective_[num_points_in_range_-1];
    
    if(thresholds_vec_[num_points_in_range_-1]>0){
      
      // This means the highest threshold value is positive
      
      if(derivative_interval.CheckIfIntervalIsNegative()){
	
	// The optimal k lies beyond the max threshold
	return ComputeGradientBeyondAndOptimalK_(num_points_in_range_-1);
      }
      else{
	
	// either the interval is greater than 0 or it has 0
	
	if(derivative_interval.CheckIfIntervalHasZero()){
	  
	  // We have found the optimal k
	  return max(0.0,thresholds_vec_[num_points_in_range_-1]);
	}
	else{
	  // The interval is positive, hence the optimal k is somewhere
	  // to the left of this threshold
	}
	
      }
    }
    else{
      
      // The max threshold is itself negative. hence the optimal k lies
      // to the right of it. Check if the derivative here has 0.
      
      if(derivative_interval.CheckIfIntervalHasZero()){
	
	return 0.0;
	
      }
      else{
	
	// Either the interval is completely positive or completely negative
	
	if(derivative_interval.CheckIfIntervalIsPositive()){
	  
	  return 0.0;
	}
	
	// The interval is negative
	
	return ComputeGradientBeyondAndOptimalK_(num_points_in_range_-1);
	
      }
    }
    
 
    // Now we have reached a point where we are guaranteed that the
    // optimal k is in the line segment defined by the thresholds
    
    double present_up;
    double next_low;
    
    for(int i=0;i<num_points_in_range_;i++){
      
      Interval derivative_interval=derivative_line_search_objective_[i];
      
      bool flag=derivative_interval.CheckIfIntervalHasZero();
      
      if(flag){
	
	// if flag=true then this interval has zero. 
	// Hence return the max of 0 and this threshold value
	
	return max(0.0,thresholds_vec_[i]);
      }
      else{
	
	// Suppose the derivative interval is completely positive.
	// This happens in a very special case, that is when the
	// derivative at the first threshold is completely positive.
	
	if(derivative_interval.CheckIfIntervalIsPositive()){
	  
	  // Hence search for optimal k before this threshold
	  
	  return ComputeGradientBeforeAndOptimalK_(i);
	}
	
	// This interval doesn't have a zero.  Obtain the upper end of
	// the interval and the lower end of the derivative at the
	// next threshold
	
	present_up=derivative_interval.up;
	next_low=derivative_line_search_objective_[i+1].low;



	Interval gradient_interval;
	gradient_interval.low=present_up;
	gradient_interval.up=next_low;

	// Now check if this test_interval has 0 	
	if(gradient_interval.CheckIfIntervalHasZero()){
	  
	  // This test interval has zero.  
	  return ComputeGradientInIntervalAndOptimalK_(i);
	}
	else{
	  // We continue 
	  
	}      
      }
    }
  }
  // If we have reached here then there is a mistake in calculations
  printf("There is an error in calculation of k_star...\n");
  exit(0);
}




void OCASLineSearch::PerformLineSearch(){

  // The first task is to calculate the thresholds for all points. 
  // Remember we are doing this calculation only for points which are in the 
  // \sigma nbhd of the query point


  CalculateThresholdsForPointsInRange_();
  SortThresholds_();

  // Now with the sorted thresholds populate the
  // derivative_line_search_objective_ array
  
  CalculateDerivatives_();

  //Finally obtain k*

  k_star_=CalculateOptimalK_();

  // Having got the k_star_ verify if this is optimal.

  if(fabs(k_star_)>SMALL){

    // This means k_star_ is not 0
    double linear_term=lambda_w_2_minus_w_1_sqd_*k_star_;

    Vector w_2_minus_w_1;
    la::SubInit(w_1_vec_, w_2_vec_, &w_2_minus_w_1);    
    double B_0=lambda_reg_const_*la::Dot(w_1_vec_,w_2_minus_w_1);

    Interval derivative_interval;
    derivative_interval.up=0.0;
    derivative_interval.low=0.0;
    double temp=0.0;
    for(int i=0;i<num_points_in_range_;i++){
      
      if(fabs(k_star_-thresholds_vec_[i])<SMALL){
	
	derivative_interval.up+=max(0.0,B_i_vec_[i]);
	derivative_interval.low+=min(0.0,B_i_vec_[i]);	
      }
      else{
	if(k_star_>thresholds_vec_[i]){
	  
	  derivative_interval.up+=max(0.0,B_i_vec_[i]);
	  derivative_interval.low+=max(0.0,B_i_vec_[i]);
	}
	else{
	  
	  derivative_interval.up+=min(0.0,B_i_vec_[i]);
	  derivative_interval.low+=min(0.0,B_i_vec_[i]);
	}
	
      }
    }
    
    //Add up all these terms

    derivative_interval.up+=linear_term+B_0;
    derivative_interval.low+=linear_term+B_0;
    
    // Finally check if the interval has 0
    int flag=derivative_interval.CheckIfIntervalHasZero();
    if(!flag){

      printf("k_star calculation is wrong....\n");
      printf("The derivative interval is [%f,%f]..\n",
	     derivative_interval.low,derivative_interval.up);
    }
  }
}

void OCASLineSearch ::Init(Matrix &train_data_appended,
			   Vector &train_labels, 
			   Vector &query_point_appended, 
			   double lambda_reg_const, 
			   ArrayList<index_t> &indices_in_range,
			   ArrayList <double> &smoothing_kernel_values_in_range,
			   int num_points_in_range, int num_train_points,
			   Vector &w_1,Vector &w_2){
  
  num_points_in_range_=num_points_in_range;

  num_train_points_=num_train_points;


  //Initialize all structures
  
  // Alias the dataset
  train_data_appended_.Alias(train_data_appended);
  
  //  train_data_appended_=train_data_appended;
  

  query_point_appended_.Alias(query_point_appended);
  //  query_point_appended_=query_point_appended;
  
  lambda_reg_const_=lambda_reg_const;

    
  // We shall initialize indices_in_range to the argument
  // provided. But note that we are copying only the values and not
  // aliasing it, because we also want to retain the unpermuted order
  // for future iterations.

  smoothing_kernel_values_in_range_.
    Copy(smoothing_kernel_values_in_range);
  
  indices_in_range_.Copy(indices_in_range);
  
  // CHANGE: Aliasing smoothing_kernel_values and indices in range
  // instead of copying them

  // smoothing_kernel_values_in_range_.InitAlias(smoothing_kernel_values_in_range);
  //indices_in_range_.InitAlias(indices_in_range);


  train_labels_.Alias(train_labels);
  
  // Initialize the thresholds
  
  thresholds_vec_.Init(num_points_in_range_);
  
  // Now initialize the derivate of line search objective
  
  derivative_line_search_objective_.Init(num_points_in_range_); 
  
  // Finally alias w_1 and w_2
  
  //w_1_vec_.Copy(w_1);
  //w_2_vec_.Copy(w_2);

  // CHANGE: Aliasing instead of copying
  w_1_vec_.Alias(w_1);
  w_2_vec_.Alias(w_2);



  // Also calculate lambda_w_2_minus_w_1_sqd

  // Get w2_minus_w1

  Vector w_2_minus_w_1;
  la::SubInit (w_1_vec_, w_2_vec_,&w_2_minus_w_1);

  lambda_w_2_minus_w_1_sqd_=
    pow(la::LengthEuclidean(w_2_minus_w_1),2)*lambda_reg_const_;

  //Initialize C_i_vec and B_i_vec

  C_i_vec_.Init(num_points_in_range_);
  B_i_vec_.Init(num_points_in_range_);
}



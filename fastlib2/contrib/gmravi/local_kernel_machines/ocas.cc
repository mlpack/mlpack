#include "fastlib/fastlib.h"
#include "utils.h"
#include "ocas.h"
#include "ocas_smo.h"
#include "ocas_line_search.h"
#define SMALL 0.0001


void OCAS::get_optimal_vector(Vector &result){
  
  result.CopyValues(w_best_at_t_);
}

void OCAS::Optimize(){
  
  
  int max_num_iterations=1;
  
  // The vector obtained by solving the quadratic program involved in
  // ocas algorithm

  Vector w_smo;
  w_smo.Init(num_dims_appended_);

  double k_star=0.0;
  
  int num_iterations=0;

  double prev_F_val=DBL_MAX;

  double prev_F_t_val=-DBL_MAX;

  double gap=DBL_MAX;

  while(num_iterations<20&&fabs(gap)>SMALL){
    
       
    // OCAS requires us to solve a QP problem where the
    // non-differentiable part of the objective is replaced by
    // subgradient. We solve this problem by calling the SMO.
    // This returns us w_at_t

    
    if(num_iterations>0)
      {
	
	// For the first iteration there is no need to solve an
	// optimization problem.

	// Firstly create an object of the type OCASSMO and initialize it
	OCASSMO ocas_smo;
	
	ocas_smo.Init(subgradients_mat_,intercepts_vec_,lambda_reg_const_);
	ocas_smo.SolveOCASSMOProblem_();
	ocas_smo.get_primal_solution(w_smo_at_t_);
	//printf("w_smo_at_t learnt is...\n");
	//w_smo_at_t_.PrintDebug();
	// This gets us w_best_at_t
	k_star=DoLineSearch_();
    
      }
    else{
      
      w_smo_at_t_.SetZero();
      k_star=0.0;
    }
    
        
    // Get w_b_at_t
    GetWBAtT_(k_star);
    double new_F_val=CalculateObjectiveFunctionValueAtAPoint_(w_best_at_t_);
    double new_F_t_val=CalculateApproximatedObjectiveValueAtAPoint_(w_best_at_t_);


    printf("prev_F_val=%f,new_F_val=%f,prev_F_t_val=%f,new_F_t_val=%f,k_star=%f",prev_F_val,new_F_val,prev_F_t_val,new_F_t_val,k_star);

    if(new_F_val-prev_F_val>pow(10,-3)||new_F_t_val-prev_F_t_val<-pow(10,-3)){
      
      printf("....Not Improved\n");
      
    }
    else{
      
      printf("...Improved\n");      
    }
   
    prev_F_val=new_F_val;
    prev_F_t_val=new_F_t_val;

    // Get approximate objective value at w_best_at_t
    double approximate_value_w_best_at_t=
      CalculateApproximatedObjectiveValueAtAPoint_(w_best_at_t_);
    
    
    //get objective value at w_best_at_t
    double objective_value_w_best_at_t=
      CalculateObjectiveFunctionValueAtAPoint_(w_best_at_t_);
    
    gap=fabs(approximate_value_w_best_at_t-objective_value_w_best_at_t);
    printf("gap is %f..\n",gap);
    printf("Current best solution is ...\n");
    for(int i=0;i<num_dims_appended_;i++){
      
      printf("%f,",w_best_at_t_[i]);
    }
    printf("\n");

    //Get the new point where the cutting plane has to be added
    GetWCAtT_();
    GetSubGradientAndInterceptAtNewPoint_();

    /*
      printf("Making a simple check...\n");
      // NOw calculate the original objective value at w_c_at_t and also
      // the approximate objective. These 2 quantities should be the
      // same
      double obj_value_w_c_at_t=
      CalculateObjectiveFunctionValueAtAPoint_(w_c_at_t_);
      
      double approx_value_w_c_at_t=CalculateApproximatedObjectiveValueAtAPoint_(w_c_at_t_);
      
      
      if(fabs(obj_value_w_c_at_t-approx_value_w_c_at_t)>pow(10,-3)){
      
      printf("obj_value=%f,approx_value=%f, ..........Seems like a mistake...\n",
      obj_value_w_c_at_t,approx_value_w_c_at_t);
      
      double val=la::Dot(subgradients_mat_[num_subgradients_available_-1],w_c_at_t_)+
      intercepts_vec_[num_subgradients_available_-1];
      printf("Expecting the contribution due to subgradient to the approximate objective function to be %f..\n",val);
      }
      else{
      
      printf("obj_value=%f,approx_value=%f, ..........Fine...\n",
      obj_value_w_c_at_t,approx_value_w_c_at_t);
      }*/
    
   
    num_iterations++;
  }

  /*
  // Get approximate objective value at w_smo_at_t
  double approximate_value_w_best_at_t=
  CalculateApproximatedObjectiveValueAtAPoint_(w_best_at_t_);
  
  
  //get objective value at w_best_at_t
  double objective_value_w_best_at_t=
  CalculateObjectiveFunctionValueAtAPoint_(w_best_at_t_);
  
  printf("obj_value=%f, approx-value=%f...\n",objective_value,approximate_value);
  
  gap=fabs(objective_value_w_best_at_t-approximate_value_w_best_at_t);
  printf("gap is %f..\n",gap);*/

  printf("finished OCAS. Required %d iterations...\n",num_iterations); 
}


void OCAS::GetWBAtT_(double k_star){
 
  if(k_star<0){

    printf("k_star is negative. Something wrong here...\n");
    exit(0);
  }
  Vector vec1;
  la::ScaleInit(1-k_star,w_best_at_t_,&vec1);

  Vector vec2;
  la::ScaleInit(k_star,w_smo_at_t_,&vec2);
  la::AddOverwrite(vec1,vec2,&w_best_at_t_);

}

void OCAS:: GetWCAtT_(){
  
  // This is simply (1-\lambda)w_best_at_t+\lambda w_t
  Vector temp1;
  la::ScaleInit(1-lambda_ocas_const_,w_best_at_t_,&temp1);

  Vector temp2;

  la::ScaleInit(lambda_ocas_const_,w_smo_at_t_,&temp2);

  la::AddOverwrite(temp1,temp2,&w_c_at_t_);
}


void OCAS::GetSubGradientAndInterceptAtNewPoint_(){

  // R(w)>=R(w_c)+<w-w_c,a>
  // R(w)>=<w,a>+R(w_c)-<w_c,a>

  // The new subgradient is -\frac{1}{n} \sum_{i=1}^n -\gamma_i y_i x_i K_i
  // where \gamma_i is a 0-1 indicator variable

  Vector new_subgradient_to_be_added;
  new_subgradient_to_be_added.Init(num_dims_appended_);
  new_subgradient_to_be_added.SetZero();
  
  for(int i=0;i<num_points_in_range_;i++){
    
    int original_index=indices_in_range_[i];
    
    double y_label=train_labels_[original_index];
    double smoothing_kernel_value=smoothing_kernel_values_in_range_[i];

    Vector x_i;
    x_i.Alias(train_data_appended_.GetColumnPtr(original_index),
	      num_dims_appended_);
    
    double w_c_at_t_dot_xi=la::Dot(w_c_at_t_,x_i);

    double val=1.0-(y_label*w_c_at_t_dot_xi);

    if(val>0||fabs(val)<SMALL){
    
      Vector scaled_x_i;
      la::ScaleInit(-1.0*y_label*smoothing_kernel_value,x_i,
		    &scaled_x_i);
      
      la::AddTo(scaled_x_i,&new_subgradient_to_be_added);
    }
  }
  // Scale the subgradient by number of train points
  la::Scale(1.0/num_train_points_,&new_subgradient_to_be_added);
  
  //  Finally calculate the intercept to be added

  double obj_value=
    CalculateNonDifferentiablePartOfObjective_(w_c_at_t_);

  double w_c_dot_subgradient=
    la::Dot(new_subgradient_to_be_added,w_c_at_t_);

  double new_intercept_to_be_added=obj_value-w_c_dot_subgradient;
  
   // Dont forget to add the new subgradient and intercept
 
  subgradients_mat_.PushBack(1);
  intercepts_vec_.PushBack(1);

  subgradients_mat_[num_subgradients_available_].Copy(new_subgradient_to_be_added);
  intercepts_vec_[num_subgradients_available_]=new_intercept_to_be_added;

  // Also dont forget to increment the number of subgradients by 1
  num_subgradients_available_++;
}


double OCAS::CalculateObjectiveFunctionValueAtAPoint_(Vector &w){


  // We need to calculate the local svm objective function value

  double lambda_by2_w_sqd;
  double length_w=la::LengthEuclidean(w);
  lambda_by2_w_sqd=
    0.5*lambda_reg_const_*length_w*length_w;
  

  double non_diff_part=CalculateNonDifferentiablePartOfObjective_(w);

  double obj_value=lambda_by2_w_sqd+non_diff_part;
  return obj_value;
}



double OCAS::CalculateNonDifferentiablePartOfObjective_(Vector &w){

  // This part is \frac{1}{n} max{1-y_i<w,x_i>,0} K_i

  double non_diff_part=0.0;
  for(int i=0;i<num_points_in_range_;i++){
    
    int original_index=indices_in_range_[i];
    
    double y_label=train_labels_[original_index];
    double smoothing_kernel_value=smoothing_kernel_values_in_range_[i];
    
    Vector x_i;
    x_i.Alias(train_data_appended_.GetColumnPtr(original_index),
	      num_dims_appended_);
    
    double w_dot_xi=la::Dot(w,x_i);
    
    double val=1-(y_label*w_dot_xi);
    non_diff_part+=max(0.0,val)*smoothing_kernel_value/num_train_points_;
  }
  return non_diff_part; 
}

double OCAS::CalculateApproximatedObjectiveValueAtAPoint_(Vector &w){

 
  double length_w=la::LengthEuclidean(w);
  double lambda_by2_w_sqd=
    0.5*lambda_reg_const_*length_w*length_w;

  double max_value=0.0;
  for(int i=0;i<num_subgradients_available_;i++){
    
    //Calculate value estimated by subgradient
    double val=la::Dot(subgradients_mat_[i],w);
    val+=intercepts_vec_[i];
    max_value=max(val,max_value);
  }
  //printf("Contribution of max of subgradients to approximate objective is %f..\n",max_value);
  return lambda_by2_w_sqd+max_value;
}

double OCAS::DoLineSearch_(){
  
  OCASLineSearch ocas_line_search;
  ocas_line_search.Init(train_data_appended_,train_labels_,
			query_point_appended_,lambda_reg_const_,
			indices_in_range_,
			smoothing_kernel_values_in_range_,
			num_points_in_range_,num_train_points_,
			w_best_at_t_,w_smo_at_t_);

  ocas_line_search.PerformLineSearch();
  return ocas_line_search.get_optimal_k();
}

void OCAS::Init(double *query_point, Matrix &train_data, 
		Vector &train_labels,
		ArrayList <int> & indices_in_range, 
		ArrayList <double> & smoothing_kernel_values_in_range, 
		double bw, double lambda_reg_const){
  
  
  // The number of train points
  
  num_train_points_=train_data.n_cols();
  
  // The number of points in the range
  
  num_points_in_range_=indices_in_range.size();

  if(indices_in_range.size()!=smoothing_kernel_values_in_range.size()){


    printf("Huge mistake...\n");
    exit(0);
  }

  //The number of dimensions after 1 has been appended to the dataset

  num_dims_appended_=train_data.n_rows();
  
  
  // Initialize query_point and set up its value.
  // Do not forget to append this quantity with a 1
 
  query_point_appended_.Init(num_dims_appended_);
  query_point_appended_.CopyValues(query_point);
  query_point[num_dims_appended_-1]=1.0;
  
  //Copy the training labels

  train_labels_.Copy(train_labels);
  
  // The training data

  train_data_appended_.Copy(train_data);
    
    
  // Initialize the other variables

  indices_in_range_.Init(num_points_in_range_);
  smoothing_kernel_values_in_range_.Init(num_points_in_range_);
  for(int i=0;i<num_points_in_range_;i++){
    
    indices_in_range_[i]=indices_in_range[i];
    smoothing_kernel_values_in_range_[i]=
      smoothing_kernel_values_in_range[i];
  }



  // The bandwidth and the regularization constant values  
  bw_smoothing_kernel_=bw;
  lambda_reg_const_=lambda_reg_const;

  // Initialize the OCAS specific quantities

  w_best_at_t_.Init(num_dims_appended_);
  w_smo_at_t_.Init(num_dims_appended_);
  w_c_at_t_.Init(num_dims_appended_);

  //Initialize w_best_at_t. Rest are calculated either by SMO or by
  //line search

  w_best_at_t_.SetZero();

 
  //Set up the subgradients and the intercepts. Always remember to
  //have the 0 subgradient
  
  subgradients_mat_.Init(1);

  // Set up the first subgradient to 0 vector

  subgradients_mat_[0].Init(num_dims_appended_);
  subgradients_mat_[0].SetZero();

  intercepts_vec_.Init(1);
  intercepts_vec_[0]=0;


  num_subgradients_available_=1;
  //Initialize lambda_ocas_constant to 0.1 as suggested in the paper

  lambda_ocas_const_=0.1;
}



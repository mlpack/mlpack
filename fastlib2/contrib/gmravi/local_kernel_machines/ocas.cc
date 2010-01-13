#include "fastlib/fastlib.h"
#include "utils.h"
#include "ocas.h"
#include "ocas_smo.h"
#include "ocas_line_search.h"

void OCAS::get_optimal_vector(Vector &result){
  
  result.CopyValues(w_best_at_t_);
}

void OCAS::Optimize(){
  
  // The vector obtained by solving the quadratic program involved in
  // ocas algorithm

  Vector w_smo;
  w_smo.Init(num_dims_appended_);

  double k_star=0.0;
  
  int num_iterations=0;

  double prev_F_t_value=0.0;

  double gap=DBL_MAX;

  Vector alpha_vec_seed;
  Vector alpha_vec_prev;

  alpha_vec_prev.Init(num_subgradients_available_);
  alpha_vec_seed.Init(num_subgradients_available_);

  double approximate_value_w_smo_at_t;
  double objective_value_w_smo_at_t;

  int MAX_NUM_ITERATIONS=200;
  while(num_iterations<MAX_NUM_ITERATIONS&&fabs(gap)>0.0001){
    
       
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
	
	ocas_smo.Init(subgradients_mat_,intercepts_vec_,lambda_reg_const_,
		      alpha_vec_seed,smoothing_kernel_bandwidth_);
	//printf("Initialized ocas smo...\n");

	ocas_smo.SolveOCASSMOProblem_();

	ocas_smo.get_primal_solution(w_smo_at_t_);
	
	ocas_smo.get_dual_solution(alpha_vec_prev);
    
	// This gets us w_best_at_t
	k_star=DoLineSearch_();
      }
    else{
      
      w_smo_at_t_.SetZero();
      alpha_vec_prev[0]=1.0;
      k_star=1.0;
    }
    
    // Get F(w_b^t-1)
    double prev_F_val_w_b_at_t=
      CalculateObjectiveFunctionValueAtAPoint_(w_best_at_t_);
    
    
    // Get w_b_at_t
    GetWBAtT_(k_star);

    // Get F(w_b^t)
    double new_F_val_w_b_at_t=
      CalculateObjectiveFunctionValueAtAPoint_(w_best_at_t_);
   
    // Note: F(w_b^t)>= F(w_b^t-1), since W_b^t is obtained by line search
    if(new_F_val_w_b_at_t-prev_F_val_w_b_at_t>SMALL){
      
      printf("Line Search did not improve on the original objective, new_F_val_w_b_at_t=%f,prev_F_val_w_b_at_t=%f...\n",
	     new_F_val_w_b_at_t,prev_F_val_w_b_at_t);
      
    }
    
    // Get approximate objective value at w_smo_at_t

    approximate_value_w_smo_at_t=
      CalculateApproximatedObjectiveValueAtAPoint_(w_smo_at_t_);
    
    
    //get objective value at w_smo_at_t

    objective_value_w_smo_at_t=
      CalculateObjectiveFunctionValueAtAPoint_(w_smo_at_t_);
   
    if(objective_value_w_smo_at_t-approximate_value_w_smo_at_t<-SMALL){
      
      printf("Minorization error...\n");
      printf("approximate value at w_smo is %f..\n",approximate_value_w_smo_at_t);
      printf("objective value at w_smo is %f..\n",objective_value_w_smo_at_t);
    }
   
    // If F represents the true objective then gap=F(w)-F_t(w)
    gap=fabs(approximate_value_w_smo_at_t-objective_value_w_smo_at_t);
    
    
    // Get F_t(w^t)
    
    double new_F_t_value=
      CalculateApproximatedObjectiveValueAtAPoint_(w_smo_at_t_);

    // Note: new_F_t_value should be better than previous F_t value

    if(new_F_t_value<prev_F_t_value-0.001){
      
      printf("Not improvement to approximate function:new_F_t_value=%f,prev_F_t_value=%f...\n",
	     new_F_t_value,prev_F_t_value);
    }
   

    prev_F_t_value=new_F_t_value;

    //Get the new point where the cutting plane has to be added

  
    // This is simply (1-\lambda)w_best_at_t+\lambda w_t
    Vector temp1;
    la::ScaleInit(1-lambda_ocas_const_,w_best_at_t_,&temp1);
    
    Vector temp2;
    
    la::ScaleInit(lambda_ocas_const_,w_smo_at_t_,&temp2);
    
    la::AddOverwrite(temp1,temp2,&w_c_at_t_);

    GetSubGradientAndInterceptAtNewPoint_(); 

    // NOw calculate the original objective value at w_c_at_t and also
    // the approximate objective. These 2 quantities should be the
    // same

    double obj_value_w_c_at_t=
      CalculateObjectiveFunctionValueAtAPoint_(w_c_at_t_);
    
    double approx_value_w_c_at_t=CalculateApproximatedObjectiveValueAtAPoint_(w_c_at_t_);
    
    // Note: The above 2 quants should match as we have added a subgradient at w_c^t
    
    if(fabs(obj_value_w_c_at_t-approx_value_w_c_at_t)>SMALL){
      
      printf("obj_value=%f,approx_value=%f, ..........Seems like a mistake...\n",
	     obj_value_w_c_at_t,approx_value_w_c_at_t);

       printf("Expecting the contribution of non_diff_part to be=%f..\n",
	      la::Dot(subgradients_mat_[num_subgradients_available_-1],w_c_at_t_)+
	      intercepts_vec_[num_subgradients_available_-1]);
    }
  
    
    // Now set up alpha_vec_seed. But first destroy the prev seed and
    // create a new one.
    
    alpha_vec_seed.Destruct();
    alpha_vec_seed.Init(num_subgradients_available_);
    
    for(int i=0;i<num_subgradients_available_-1;i++){
      if(i==0){
	alpha_vec_seed[i]=alpha_vec_prev[i];
      }
      else{

	alpha_vec_seed[i]=alpha_vec_prev[i];
      }
      //alpha_vec_seed[i]=alpha_vec_prev[i];
    }
    alpha_vec_seed[num_subgradients_available_-1]=0.0;

    
    // Now destroy previous alpha vector and initialize it. 
    // This is because its size has changed
    
    alpha_vec_prev.Destruct();
    alpha_vec_prev.Init(num_subgradients_available_);
    
    num_iterations++;
      // printf("NUmber of iterations=%d..\n",num_iterations);
  }
  
  printf("Finally at the end of OCAS we have...\n");
  printf("approximate_value_w_smo_at_t=%f...\n",approximate_value_w_smo_at_t);
  printf("objective_value_w_smo_at_t=%f..\n",objective_value_w_smo_at_t);
  printf("gap=%f...\n",gap);

  if(num_iterations==MAX_NUM_ITERATIONS){

    printf("CAUTION:OCAS Required %d iterations...\n",num_iterations); 
    printf("lambda=%f,smoothing_kernel_bandwidth=%f..\n",
	   lambda_reg_const_,smoothing_kernel_bandwidth_);
    printf("gap=%f..\n",gap);
  }
  else{
    printf("Number of OCAS iterations=%d..\n",num_iterations);

  }
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


void OCAS::PrintSubgradientsAndIntercepts_(){
  
  for(int i=0;i<num_subgradients_available_;i++){

    printf("Subgradient is:");
    for(int j=0;j<num_dims_appended_;j++){

      printf("%f,",subgradients_mat_[i][j]);
    }
    printf("...\n");
  }

  printf("The intercepts are...\n");
  for(int i=0;i<num_subgradients_available_;i++){

    printf("%f,",intercepts_vec_[i]);
  }
}

void OCAS::GetSubGradientAndInterceptAtNewPoint_(){

  // R(w)>=R(w_c)+<w-w_c,a>
  // R(w)>=<w,a>+R(w_c)-<w_c,a>

  // The new subgradient is -\frac{1}{n} \sum_{i=1}^n -\gamma_i y_i x_i K_i
  // where \gamma_i is a 0-1 indicator variable
  
  Vector subgradient_to_be_added;
  subgradient_to_be_added.Init(num_dims_appended_);
  subgradient_to_be_added.SetZero();
  
  for(int z=0;z<indices_in_range_.size();z++){
    
    Vector x_i;
    int original_pos=indices_in_range_[z];
    x_i.Alias(train_data_appended_.GetColumnPtr(original_pos),
	      num_dims_appended_);

    double y_lab=train_labels_[original_pos];
    double w_c_dot_x_i=la::Dot(w_c_at_t_,x_i);
    double val=1-y_lab*(w_c_dot_x_i);

    double smoothing_kernel_value=
      smoothing_kernel_values_in_range_[z];


    if(val>0 || fabs(val)<SMALL){
      
      Vector temp;
      la::ScaleInit(-1*y_lab*smoothing_kernel_value/num_train_points_
		    ,x_i,&temp);
	
      la::AddTo(temp,&subgradient_to_be_added);

    } 
  }
 
  //  Finally calculate the intercept to be added

  
  double intercept_to_be_added=0.0;
  for(int j=0;j<smoothing_kernel_values_in_range_.size();j++){
    
    int original_pos;
    original_pos=indices_in_range_[j];
    
    double y_lab=train_labels_[original_pos];

    Vector x_i;
    x_i.Alias(train_data_appended_.GetColumnPtr(original_pos),
	      num_dims_appended_);

    double val=1-(y_lab*la::Dot(w_c_at_t_,x_i));

    double pi;
    if(val>0||fabs(val)<SMALL){
      
      pi=1.0;
      
    }
    else{
      pi=0.0;
    }

    intercept_to_be_added+=
      pi*smoothing_kernel_values_in_range_[j];
  }
  intercept_to_be_added/=num_train_points_;
  
  if(intercept_to_be_added<0){
    
    printf("The intercept=%f came out to be negative...\n",
	   intercept_to_be_added);
    exit(0);
  }
   // Dont forget to add the new subgradient and intercept
  
  subgradients_mat_.PushBack(1);
  intercepts_vec_.PushBack(1);
  
  subgradients_mat_[num_subgradients_available_].
    Copy(subgradient_to_be_added);
  
  intercepts_vec_[num_subgradients_available_]=intercept_to_be_added;
  
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

  // printf("Contribution of non-diff part to the objective=%f..\n",non_diff_part);
  return obj_value;
}


double OCAS::CalculateNonDifferentiablePartOfObjective_(Vector &w){
  
  // This part is \frac{1}{n} \sum max{1-y_i<w,x_i>,0} K_i
  
  double non_diff_part=0.0;

  for(int i=0;i<indices_in_range_.size();i++){

    int original_pos=indices_in_range_[i];
    double y_lab=train_labels_[original_pos];

    Vector x_i;
    x_i.Alias(train_data_appended_.GetColumnPtr(original_pos),num_dims_appended_);
    
    double kernel_value=smoothing_kernel_values_in_range_[i];
    double w_dot_x_i=la::Dot(w,x_i);
    double val=1-y_lab*w_dot_x_i;

    if(val>0){
      
      non_diff_part+=val*kernel_value/num_train_points_;
    }
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
  
  // printf("Contribution of non-diff part using max of subgradients to approximate objective=%f..\n",max_value);
   
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
  
  //query_point_appended_.Init(num_dims_appended_);
  //query_point_appended_.CopyValues(query_point);
  
  // CHANGE: ALIASING the query
  query_point_appended_.Alias(query_point,num_dims_appended_);

  //Alias the training labels

  train_labels_.Alias(train_labels);
  
  // Alias The training data

  train_data_appended_.Alias(train_data);
    
    
  // Alias the other variables

  // CHANGE: InitAliasing instead of element wise copying

  indices_in_range_.InitAlias(indices_in_range);
  smoothing_kernel_values_in_range_.InitAlias(smoothing_kernel_values_in_range);

  // indices_in_range_.Init(num_points_in_range_);
//   smoothing_kernel_values_in_range_.Init(num_points_in_range_);
//   for(int i=0;i<num_points_in_range_;i++){
    
//     indices_in_range_[i]=indices_in_range[i];
//     smoothing_kernel_values_in_range_[i]=
//       smoothing_kernel_values_in_range[i];
//   }

  
  
  // The bandwidth and the regularization constant values  
  smoothing_kernel_bandwidth_=bw;
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



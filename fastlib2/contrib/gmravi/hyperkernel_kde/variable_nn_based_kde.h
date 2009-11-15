/** This piece of code does LSCV for variable bandwidth kde, with the
    bandiwdth of each reference point picked by using NN
    distances. Once the optimal parameters are found then finally the
    optimal parameters are used to calculate the density estimates at
    the different test points
*/


#include "fastlib/fastlib.h"
#include "mlpack/kde/naive_kde.h"
#include "mlpack/kde/dualtree_vkde.h"
#include "mlpack/fastica/lin_alg.h"
#define EPSILON 0.00001
#define LEAF_SIZE 5
class VariableNNKDE{
  
 private:


  //The train and test datasets
  Matrix train_data_;

  Matrix test_data_;

  //The true densities at the test points(This may be available in
  //case of synthetic distributions)

  Matrix true_test_densities_;

  //Vector of bandwidths of the different train points

  Vector bandwidth_train_points_;

  //The value of k_opt

  index_t k_opt_;

  //The universal smoothing factor

  double h_opt_;

  //The plugin bandwidth. We will use this value to base our search
  //for the parameter h, which is the global smoothing factor.
  
  double h_plugin_;

  //A flag to tell if the true density file is present

  index_t true_density_file_present_;

  //The number of train and test points

  index_t num_train_points_;
  index_t num_test_points_;

  //The dimensionality of the dataset
  index_t num_dims_;

  //The vector of densities at different test points

  Vector test_densities_;

  //module to hold the incoming module

  struct datanode *module_;

  //Flag to say if the test file is present or not

  index_t test_file_present_;

  //Gaussian kernel that we shall use everywhere in the code

  GaussianKernel gk_;

  //Max number of nearest neighbours to be considered

  index_t  max_number_of_neighbours_;
    
  //Similarily number of global smoothing values to be considered
    
  index_t number_of_global_smoothing_values_;


 public:
  
  //A bunch of getter functions
  /*This function calculates the rmse iff true test densities are known */
  void get_rmse_and_hellinger_distance(double *rmse,double *hellinger_distance){
    
    index_t len=true_test_densities_.n_cols();
    
    if(len==0){
      
      *rmse=-1;
      *hellinger_distance=-10;
    }
    
    double diff=0;
    double total_sqd_diff=0;
    double sum_ratio=0;
    for(index_t i=0;i<len;i++){
      
      diff=test_densities_[i]-true_test_densities_.get(0,i);
      total_sqd_diff+=diff*diff;

      double sqrt_ratio=
	sqrt(test_densities_[i]/true_test_densities_.get(0,i));
      sum_ratio+=sqrt_ratio;
    }
    *rmse=sqrt(total_sqd_diff/num_test_points_);
    *hellinger_distance=2-(2*sum_ratio/num_test_points_); 
  }

  void get_test_densities(Vector *result){

    printf("Test densities are..\n");
    test_densities_.PrintDebug();
    printf("Initialized with %d points",num_test_points_);
    result->Init(num_test_points_);
   
    result->CopyValues(test_densities_);
  }


 private:
    
 
  
  //This function fills the vector of bandwidths of the different train
  //points.

  //The param $h$ measures the global smoothing and 

  void FillVectorOfBandwidths_(double h,index_t k){
    
    // Initialize the kernels for each reference point.
    
    AllkNN all_knn;
    all_knn.Init(train_data_,LEAF_SIZE,k);
    ArrayList<index_t> resulting_neighbors;
    ArrayList<double> squared_distances;    
    
    all_knn.ComputeNeighbors(&resulting_neighbors, &squared_distances);

    for(index_t i = 0; i < squared_distances.size(); i += k) 
      {
	
	bandwidth_train_points_[i / k]=
	  h*sqrt(squared_distances[i+k-1]);
      }
  }


  void CalculateMean_(Matrix &data,Vector &mean){
    
    index_t num_points=data.n_cols();
    index_t num_dim=data.n_rows();
    mean.SetZero();
    
    for(index_t i=0;i<num_points;i++){
      
      //Get the present data point
      double *data_point=data.GetColumnPtr(i);
      Vector dummy;
      dummy.Copy(data_point,num_dim);
      la::AddTo(dummy,&mean);  
    }
    
    //Finally divide by the total number of data points
    
    la::Scale(1.0/num_points,&mean);
  }

  /*This function returns a vector with all its elements squared */

  void GetSquaredVector_(Vector &vec){
    
    for(index_t i=0;i<num_dims_;i++){

      double elem=vec[i];
      vec[i]=elem*elem;
    }
  }
  
  /*This function calculates the standard deviation of the dataset*/
  
  double CalculateMarginalVariance_(Matrix &data){
    
    //First calculate the mean
    Vector mean;
    mean.Init(num_dims_);
    CalculateMean_(data,mean);
    
    //We now have the mean. We can easily calculate the standard deviation

   
    //A doubl;e pointer to the mean
   
    Vector sum;
    sum.Init(num_dims_);
    sum.SetZero();

    Vector diff;
    diff.Init(num_dims_);
    for(index_t i=0;i<num_train_points_;i++){

      double *present_point_ptr;
      present_point_ptr=data.GetColumnPtr(i);

      Vector present_point;
      present_point.Alias(present_point_ptr,num_dims_);
      
      la::SubOverwrite(present_point,mean,&diff);
      GetSquaredVector_(diff);
      la::AddTo(diff,&sum);
    }

    la::Scale(1.0/num_train_points_,&sum);

    double total_val=0;
    for(index_t i=0;i<num_dims_;i++){

      total_val+=sum[i];
    }

    return sqrt(total_val/num_dims_);

  }

  void FindPluginBandwidth_(){

    //h_plugin=A(K) n^-(\frac{1}{d+4})
    
    //We will use a multiplicative gaussian kernel
    
    //First calculate the standard deviation of train data
    double std=CalculateMarginalVariance_(train_data_);
    double A_k=pow(4.0/(num_dims_+2),1.0/(num_dims_+4));
    h_plugin_=std*A_k*pow(num_train_points_,-1.0/(num_dims_+4));  

    //It is usually suggested to take a value smaller than the
    //calculated plugin bandwidth. We shall arbitrarily take it as
    //4/5th of the value

    h_plugin_*=0.8;
    
    printf("plugin bandwidth is %f..\n",h_plugin_);
  }
 

  double CalculateIntegralFHatSqd_(){
  
    //\int \hat{f}^2=\frac{1}[n^2}\sum_[i=1}^n\sum_{j=1}^n
    //K_{\sqrt{h_i^2+h_j^2}}\left(x_i-x_j\right)

    double int_f_hat_sqd=0;
    GaussianKernel gk;
    
    for(index_t i=0;i<num_train_points_;i++){

      for(index_t j=i;j<num_train_points_;j++){
	
	double bw_i=bandwidth_train_points_[i];
	double bw_j=bandwidth_train_points_[j];

	double eff_bw=sqrt((bw_i*bw_i)+(bw_j*bw_j));

	//Initialize the gaussian kernel with this bandwidth

	gk.Init(eff_bw);

	double norm_const=
	  gk.CalcNormConstant(num_dims_);

	double *x_i=train_data_.GetColumnPtr(i);
	double *x_j=train_data_.GetColumnPtr(j);

	double sqd_dist=
	  la::DistanceSqEuclidean(num_dims_, x_i, x_j);

	double unnorm_kernel_val=
	  gk.EvalUnnormOnSq(sqd_dist);

	
	double val=(unnorm_kernel_val)/norm_const;

	if(i!=j){
	  int_f_hat_sqd+=2*val;
	}
	else{
	  int_f_hat_sqd+=val;
	}
      }
    }
    return int_f_hat_sqd/(num_train_points_*num_train_points_);
  }
  
 

  double CalculateIntegralFF_hat(){

    double integral_f_f_hat=0;
    GaussianKernel gk;
    
    for(index_t i=0;i<num_train_points_;i++){
      
      for(index_t j=0;j<num_train_points_;j++){
	
	if(j!=i){

	  double bw_j=bandwidth_train_points_[j];
	  
	  //Initialize the gaussian kernel with this bandwidth
	  
	  gk.Init(bw_j);
	  
	  double *x_i=train_data_.GetColumnPtr(i);
	  double *x_j=train_data_.GetColumnPtr(j);
	  double sqd_dist=
	    la::DistanceSqEuclidean(num_dims_, x_i, x_j);
	  
	  double unnorm_kernel_val=
	    gk.EvalUnnormOnSq(sqd_dist);
	  
	  double norm_const=
	    gk.CalcNormConstant(num_dims_);
	  
	  double val=(unnorm_kernel_val)/norm_const;

	  integral_f_f_hat+=val;
	}
	else{
	  //We wont take this contribution in because of leave one out estimate
	}
      }
    }

    return integral_f_f_hat/(num_train_points_*(num_train_points_-1));    
  }


  void FillGlobalSmoothingVector_(Vector &global_smoothing_vec){

    //This piece of code needs to change
    
    //We shall fill the possible gobal smoothing values by first
    //finding the plugin bandwidth.

    FindPluginBandwidth_();
    
    double min_bw=h_plugin_*0.5;
    double max_bw=5*h_plugin_;
    double gap=(max_bw-min_bw)/number_of_global_smoothing_values_;
    
    for(index_t i=0;i<number_of_global_smoothing_values_;i++){
      
      global_smoothing_vec[i]=min_bw+i*gap;
    }
   
  }

void WhitenData_(){

      Matrix train_data_whitened;
      Matrix test_data_whitened;
      
      Matrix whitening_matrix_train;
      Matrix whitening_matrix_test;
      
      index_t num_dims=train_data_.n_rows();
      if(num_dims>1){
	
	//Whiten the data. This allows us to use a single bandiwdth
	//parameter in all directions.
	
	linalg__private::WhitenUsingEig(train_data_,&train_data_whitened,
					&whitening_matrix_train);
	
	linalg__private::WhitenUsingEig(test_data_,&test_data_whitened,
					&whitening_matrix_test);  	
	//Now alias it back
	train_data_.CopyValues(train_data_whitened);
	test_data_.CopyValues(test_data_whitened);
	
	printf("dataset was whitened..\n");
	printf("test data is..\n");
	printf("Num dims =%d\n",test_data_.n_rows());
	printf("Num points=%d..\n",test_data_.n_cols());
	FILE *fp=fopen("old_faithful_whitened_test.txt","w+");
	index_t num_points=test_data_.n_cols();
	index_t num_dims=test_data_.n_rows();

	for(index_t n=0;n<num_points;n++){

	  fprintf(fp,"%f,%f\n",test_data_.get(0,n),test_data_.get(1,n));
	}
	test_data_.PrintDebug();
      }
      else{
	
	//whitening was not done
	train_data_whitened.Init(1,1);
	test_data_whitened.Init(1,1);
	whitening_matrix_train.Init(1,1);
	whitening_matrix_test.Init(1,1);
      }
  }
  
 public:


 void ComputeTestDensities(){

   //Whiten data if required
   //WhitenData_();

    //before we calculate the test densities, lets calculate the
    //bandwidths of the train points

   Matrix rset_weights; 
   rset_weights.Init(1,num_test_points_); 
   for(index_t i=0;i<rset_weights.n_rows();i++){ 
     
     for(index_t j=0;j<rset_weights.n_cols();j++){ 
       
       rset_weights.set(i,j,1); 
     } 
   } 
   
    NaiveKde<GaussianKernel> naive_variable_kde;
    struct datanode *naive_variable_kde_module;
    naive_variable_kde_module = 
      fx_submodule(NULL, "naive_variable_kde_module");
    
    //Set some parameters of the module

    printf("k_opt=%d...\n",k_opt_);
    printf("h_opt=%f...\n",h_opt_);
    fx_set_param_int(naive_variable_kde_module,"knn",k_opt_);
    fx_set_param_double(naive_variable_kde_module,"global_smoothing",h_opt_);
    fx_param_str(naive_variable_kde_module, "mode", "variablebw");

    naive_variable_kde.Init(test_data_,train_data_,
			    rset_weights,naive_variable_kde_module);

    naive_variable_kde.Compute(&test_densities_);
    printf("test densities are...\n");
    test_densities_.PrintDebug();
 }
  
  //This peice of code does LSCV to find the appropriate values for
  //$k$ and $h_opt$
  
  void PerformLSCV(){
    
    Vector global_smoothing_vec;
    global_smoothing_vec.Init(number_of_global_smoothing_values_);
    
    //Fill the global smoothing vector
    FillGlobalSmoothingVector_(global_smoothing_vec);
    
    double min_lscv_score=DBL_MAX;

    //This step fills a vector of bandwidths to be used for all the
    //train points
       
    for(index_t k=3;k<=max_number_of_neighbours_;k++){
      
      for(index_t j=0;j<global_smoothing_vec.length();j++){
	
	FillVectorOfBandwidths_(global_smoothing_vec[j],k);
	
	
	double int_f_hat_sqd=
	  CalculateIntegralFHatSqd_();
	
	double int_f_f_hat=CalculateIntegralFF_hat();

	//cvs=int_f_hat_sqd-2*int_f_f_hat
	
	double lscv_score= int_f_hat_sqd-2*int_f_f_hat;
	
	if(lscv_score<min_lscv_score){

	  //Found a better combination
	  h_opt_=global_smoothing_vec[j];
	  k_opt_=k;

	  //Change the min_lscv_score
	  
	  min_lscv_score=lscv_score;
	}
	else{
	  
	  //This is a worser combination. Do nothing
	}
      }
    }
    printf("The optimal settings are....\n");
    printf("Number of neighbours including self are %d...\n",k_opt_);
    printf("global smoothing value is %f...\n",h_opt_);
    printf("The LSCV score is %f\n",min_lscv_score);
  }
  

  void Init(struct datanode *module_in){
    //Copy the module
    
    module_=module_in;
    
    //Train data file is a requirement
    const char *train_file=
      fx_param_str_req(module_,"train");
    
    //Load the train dataset
    
    data::Load(train_file,&train_data_);
    num_train_points_=train_data_.n_cols();
    
    
    //Variables concerning test set

     num_test_points_=0;
     true_density_file_present_=0;

     //The input dimensionality    
     num_dims_=train_data_.n_rows();
    
    
    //Check for the existence of a test set and accordingly read data
    if(fx_param_exists(module_,"test")){
      
      //Load the dataset
      const char *test_file=fx_param_str_req(module_,"test");
      data::Load(test_file,&test_data_);
      test_file_present_=1;
      num_test_points_=test_data_.n_cols();
    }
    else{
      test_file_present_=0;
      test_data_.Init(0,0); //This avoids segmentation fault
      num_test_points_=0;
    }

    if(fx_param_exists(module_,"true")){

      true_density_file_present_=1;      
      const char *true_density_file=fx_param_str_req(module_,"true");
      //Since the true test densities are given, hence
      data::Load(true_density_file,&true_test_densities_);
    }
    else{
      
      //The true test density file is not present

      true_density_file_present_=0;
      //To avoid seg fault initialize it

      true_test_densities_.Init(0,0);      
    }
    
    
    //Initialize other  necessary vectors
    
    test_densities_.Init(num_test_points_); 
    bandwidth_train_points_.Init(num_train_points_);


    //The max number of neighbours and the number of global smoothing
    //values to be considered

    max_number_of_neighbours_=fx_param_int(module_,"max_number_of_neighbours",num_train_points_/10);
    printf("Number of neighbours=%d..\n",max_number_of_neighbours_);
    number_of_global_smoothing_values_=fx_param_int(module_,"number_of_global_smoothing_values",50);

    printf("Succesfully initializesd...\n");

  }
  
};

#include "fastlib/fastlib.h"
#include "mlpack/kde/naive_kde.h"
#define EPSILON 0.00001
class AdaptiveKDE{
  
 private:

  //Vector of bandwidths
  Vector bandwidth_train_points_;
  
  //Vector of densities's at all test points
  Vector test_densities_;

  //The pilot bandwidth to be used
  double pilot_bw_;
  

  //Tbe train set
  Matrix train_data_;
  Matrix test_data_;

  //The true density values if provided

  Matrix true_test_density_data_;

  //module to hold the incoming module

  struct datanode *module_;

  //Number of training points

  index_t num_train_points_;

  //Number of test points
  index_t num_test_points_;

  index_t num_dims_;

  double plugin_bandwidth_;

  double geometric_mean_;

  double alpha_;

 public:

  //A bunch of getter functions...

  //Get the test_densities

  void get_test_densities(Vector &results){

    results.Alias(test_densities_);
  }

  
  /*This function calculates the rmse iff true test densities are known */
  double get_rmse(){
    
    index_t len=true_test_density_data_.n_cols();

    if(len==0){

      return -1.0;
    }

    double diff=0;
    double total_sqd_diff=0;
    for(index_t i=0;i<len;i++){

      diff=test_densities_[i]-true_test_density_data_.get(0,i);
      total_sqd_diff+=diff*diff;
    }
    return sqrt(total_sqd_diff/num_test_points_); 
  }

 private:

  /*This function calculates the mean of the dataset */

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
    plugin_bandwidth_=std*A_k*pow(num_train_points_,-1.0/(num_dims_+4));  

    //It is usually suggested to take a value smaller than the
    //calculated plugin bandwidth. We shall arbitrarily take it as
    //4/5th of the value

    plugin_bandwidth_*=0.8;
    
    printf("plugin bandwidth is %f..\n",plugin_bandwidth_);
    printf("PLUGIN BANDWIDTH CALCULATION STILL NOT COMPLETE..\n"); 
  }
  

  void MakeEstimatesUnbiased_(Vector &train_densities){

    double norm_const;
    GaussianKernel gk;
    gk.Init(plugin_bandwidth_);
    norm_const=gk.CalcNormConstant(num_dims_);

    double const1=1.0/((num_train_points_-1)*norm_const);

    for(index_t i=0;i<num_train_points_;i++){
      
      double initial_density=train_densities[i];
      train_densities[i]=
	(num_train_points_*initial_density/(num_train_points_-1))-const1;
    } 
  }

  void ComputePilotDensitiesUsingPluginBandwidth_(Vector &train_densities){
    
    //We shall use MLPack's naive kde code to do these calculations.
    //First create a module called naive

    struct datanode *naive=fx_submodule(NULL,"naive");;
    NaiveKde<GaussianKernel> naive_kde;

    fx_set_param_double(naive,"bandwidth", plugin_bandwidth_);
    fx_set_param_str(naive,"mode","fixedbw");
    
    naive_kde.Init(train_data_,train_data_,naive);
    naive_kde.Compute();
    naive_kde.get_density_estimates(&train_densities);

    //remember the estimate so obtained is biased, since
    //train_set=test_set. Hence we shall make it ubiased by
    //subtracting the self-contribution

    MakeEstimatesUnbiased_(train_densities);
  }

  void CalculateGM_(Vector &train_densities){
    
    double log_gm=0;
    double likelihood=1;
    for(index_t i=0;i<num_train_points_;i++){

      likelihood*=train_densities[i];
      
      if(likelihood<EPSILON){
	//The likelihood has become very small. To prevent underflow
	//lets take logarithm and add
	log_gm+=log(likelihood);
	likelihood=1;
      }
    }
    log_gm+=log(likelihood);
    geometric_mean_=pow(10,log_gm/num_train_points_);
  }
  
  void CalculateLocalBandwidths_(Vector &train_densities){

    for(index_t i=0;i<num_train_points_;i++){
      
      double lambda=pow(train_densities[i]/geometric_mean_,-alpha_);
      bandwidth_train_points_[i]=plugin_bandwidth_*lambda;
    }

    printf("Bandwidth of train points is...\n");
    bandwidth_train_points_.PrintDebug();
  }

  void FindPilotDensities_(Vector &train_densities){
    
    //We first compute the pilot density estimates at all test points.
    //It has been empirically observed that the method of getting
    //pilot density estimates doesn't influence further calculations
    //We shall use fixed bandwidth kde with plugin bandwidth.   
    
    
    FindPluginBandwidth_(); 
    ComputePilotDensitiesUsingPluginBandwidth_(train_densities);
  }

  void CalculateTestDensities_(){
    
    GaussianKernel gk;
    
    for(index_t i=0;i<num_test_points_;i++){
    
      double *test_point=test_data_.GetColumnPtr(i);
      double total_contrib=0;
      for(index_t j=0;j<num_train_points_;j++){  

	//get the contribution of the jth trg point
	gk.Init(bandwidth_train_points_[j]);
	double norm_const=gk.CalcNormConstant(num_dims_);

	double *train_point=train_data_.GetColumnPtr(j);
	double sqd_dist=
	  la::DistanceSqEuclidean (num_dims_,train_point,test_point);
	double contrib=gk.EvalUnnormOnSq(sqd_dist);
	contrib/=norm_const;
	total_contrib+=contrib;
      }
      //Store the density calculated
      test_densities_[i]=total_contrib/num_train_points_;
    }
    printf("Priniting test densities obtained by adaptive kde...\n");
    test_densities_.PrintDebug();
  }

 public:
  
  void ComputeDensities(){

    Vector train_densities;
    train_densities.Init(num_train_points_);

    FindPilotDensities_(train_densities);
    CalculateGM_(train_densities);
    CalculateLocalBandwidths_(train_densities);
    
    CalculateTestDensities_();
  }

  void Init(struct datanode *module_in){

    //Hold the incoming module
    module_=module_in;

    const char *train_file=fx_param_str_req(module_,"train");
    data::Load(train_file,&train_data_);
    
    //Get the test file too
    bool test_file_present=fx_param_exists(module_,"test");
    
    if(test_file_present==1){
      
      const char *test_file=fx_param_str_req(module_,"test");
      //Load the test data 
      
      data::Load(test_file,&test_data_);
    }
    else{
      //The test file is not given. To avoid segmentation fault
      //inititalize to somne small memory
      
      test_data_.Init(0,0);
    }


    //Get the true densities if they are given 

    bool true_density_file_present=fx_param_exists(module_,"true");
    
    if(true_density_file_present==1){
      
      const char *true_density_file=fx_param_str_req(module_,"true");
      //Load the true density file 
      
      data::Load(true_density_file,&true_test_density_data_);
    }
    else{
      //The test file is not given. To avoid segmentation fault
      //inititalize to somne small memory
      
      true_test_density_data_.Init(0,0);
    }
    
    //Number of train points,test points and the dimensionality
    num_train_points_=train_data_.n_cols();
    num_dims_=train_data_.n_rows();
    num_test_points_=test_data_.n_cols();

    //Initialize the test densities set
    test_densities_.Init(num_test_points_);

    //Initialize bandwidth_train_points_
    bandwidth_train_points_.Init(num_train_points_);

    //Set the  value of alpha

    alpha_=fx_param_double(module_,"alpha",0.5);
  }
};


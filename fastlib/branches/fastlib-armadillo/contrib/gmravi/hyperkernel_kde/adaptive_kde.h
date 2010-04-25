#include "fastlib/fastlib.h"
#include "mlpack/kde/naive_kde.h"
#include "mlpack/fastica/lin_alg.h"
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

  //The train and test labels

  Matrix train_labels_;
  Matrix test_labels_;
  
  //module to hold the incoming module

  struct datanode *module_;

  //Number of training points

  index_t num_train_points_;

  //Number of test points
  index_t num_test_points_;

  index_t num_dims_;

  index_t classification_flag_;

  double plugin_bandwidth_;

  double geometric_mean_;

  double alpha_;

 public:

  //A bunch of getter functions...

  //Get the test_densities

  void get_test_densities(Vector &results){

    results.Copy(test_densities_);
    printf("Test densities have come out to be...\n");
    test_densities_.PrintDebug();

  }

  
  /*This function calculates the rmse iff true test densities are known */
  void get_rmse_and_hellinger_distance(double *rmse_adaptive_kde,
				       double *hellinger_adaptive_kde){
    
    index_t len=true_test_density_data_.n_cols();

    if(len==0){

      *rmse_adaptive_kde=-1.0;
      *hellinger_adaptive_kde=-10000.0;
      return;
    }

    double diff=0;
    double total_sqd_diff=0;
    double sum_ratio=0;
    for(index_t i=0;i<len;i++){

      diff=test_densities_[i]-true_test_density_data_.get(0,i);
      total_sqd_diff+=(diff*diff);
      double sqrt_ratio=
	sqrt(test_densities_[i]/true_test_density_data_.get(0,i));
      sum_ratio+=sqrt_ratio;
    }
    
    *hellinger_adaptive_kde=
      2-(2*sum_ratio/num_test_points_);
    *rmse_adaptive_kde =sqrt(total_sqd_diff/num_test_points_);
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
    
    printf("Train densities are..\n");
    train_densities.PrintDebug();
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
    printf("geometric_mean is %f..\n",geometric_mean_);
  }
  
  void CalculateLocalBandwidths_(Vector &train_densities){

    for(index_t i=0;i<num_train_points_;i++){
      
      double lambda=pow(train_densities[i]/geometric_mean_,-alpha_);
      //   printf("lambda is %f...\n",lambda);
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


  void CalculateClassificationAccuracy_(){

    GaussianKernel gk;
    
    index_t num_correctly_classified=0;
    for(index_t i=0;i<num_test_points_;i++){
    
      double *test_point=test_data_.GetColumnPtr(i);
      double classification_contrib=0;
      for(index_t j=0;j<num_train_points_;j++){  

	//get the contribution of the jth trg point
	gk.Init(bandwidth_train_points_[j]);
	double *train_point=train_data_.GetColumnPtr(j);
	double sqd_dist=
	  la::DistanceSqEuclidean (num_dims_,train_point,test_point);
	double contrib=gk.EvalUnnormOnSq(sqd_dist);
	
	classification_contrib+=(contrib*train_labels_.get(j,0));
      }
      //      printf("classification_contrib=%f..\n",classification_contrib);
      if(classification_contrib*test_labels_.get(i,0)>=0){

	num_correctly_classified++;
      }
      //Store the density calculated
    }
    printf("Classification Accuracy =%f..\n",(double)num_correctly_classified/num_test_points_);

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
	

	//Before that center the data

	
	linalg__private::WhitenUsingSVD(train_data_,&train_data_whitened,
					&whitening_matrix_train);
	
	linalg__private::WhitenUsingSVD(test_data_,&test_data_whitened,
					&whitening_matrix_test);  	
	//Now alias it back
	train_data_.CopyValues(train_data_whitened);
	test_data_.CopyValues(test_data_whitened);
	
	printf("dataset was whitened..\n");
	printf("test data is..\n");
	printf("Num dims =%d\n",test_data_.n_rows());
	printf("Num points=%d..\n",test_data_.n_cols());
	FILE *fp=fopen("/net/hc295/gmravi/home/research/qp_and_boosted_kde/banana/banana_whitened_test.txt","w+");
	index_t num_points=test_data_.n_cols();
	index_t num_dims=test_data_.n_rows();

	for(index_t n=0;n<num_points;n++){

	  fprintf(fp,"%f,%f\n",test_data_whitened.get(0,n),test_data_whitened.get(1,n));
	}
	

	FILE *gp=fopen("/net/hc295/gmravi/home/research/qp_and_boosted_kde/banana/banana_whitened_train.txt","w+");
	index_t num_train_points=train_data_.n_cols();
	
	for(index_t n=0;n<num_train_points;n++){

	  fprintf(gp,"%f,%f\n",train_data_whitened.get(0,n),train_data_whitened.get(1,n));
	}
      }
      else{
	
	//whitening was not done
	train_data_whitened.Init(1,1);
	test_data_whitened.Init(1,1);
	whitening_matrix_train.Init(1,1);
	whitening_matrix_test.Init(1,1);
      }

      //Write back the whitened train and test datas
      
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

 double CalculateIntegralFFHat_(){

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

  
  
  void CalculateLSCVScore_(){
    
    double int_f_hat_sqd=CalculateIntegralFHatSqd_();
    double int_f_f_hat=CalculateIntegralFFHat_();
    double lscv_score=int_f_hat_sqd-2*int_f_f_hat;
    printf("The LSCV score is %f..\n",lscv_score);
  }


  
 public:
  
  void ComputeDensities(){


    //Even before u start computations whiten the data

    // WhitenData_();
    printf("Whitened the dataset......\n");

    Vector train_densities;
    train_densities.Init(num_train_points_);

    FindPilotDensities_(train_densities);
    printf("Found pilot densities...\n");
    CalculateGM_(train_densities);
    printf("Calculated GM...\n");
    CalculateLocalBandwidths_(train_densities);
    printf("Calculated local bandwidths.....\n");

    printf("The classification flag is set to %d...\n",classification_flag_);

    
    CalculateLSCVScore_();
    CalculateTestDensities_();

    printf("Calculated LSCV score..;....\n");
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
    printf("Test data is....\n");
    test_data_.PrintDebug();

    printf("Train data is...\n");
    train_data_.PrintDebug();

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



    bool class_file_present=fx_param_exists(module_,"train_labels");
    
    if(class_file_present==1){

      printf("Train classes are given. It is a classification problem...\n");

      const char *train_labels_file=fx_param_str_req(module_,"train_labels");

      const char *test_labels_file=fx_param_str_req(module_,"test_labels");

      data::Load(train_labels_file,&train_labels_);
      data::Load(test_labels_file,&test_labels_);

      classification_flag_=1;
    }

    else{
      train_labels_.Init(1,1);
      test_labels_.Init(1,1);
      classification_flag_=0;
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


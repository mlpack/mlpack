/** This piece of code does LSCV for variable bandwidth kde, with the
    bandiwdth of each reference point picked by using NN
    distances. Once the optimal parameters are found then finally the
    optimal parameters are used to calculate the density estimates at
    the different test points
*/


#include "fastlib/fastlib.h"
#include "mlpack/kde/naive_kde.h"
#include "mlpack/kde/dualtree_vkde.h"
#define EPSILON 0.00001
#define LEAF_SIZE 3
class NNKDE{
  
 private:


  //The train and test datasets
  Matrix train_data_;

  Matrix test_data_;

  //The true densities at the test points(This may be available in
  //case of synthetic distributions)

  Matrix true_test_densities_;
 

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

  index_t num_of_neighbours_;
  
  //Vector of bandwidths of test points

  Vector bandwidth_test_points_;



 public:

 //A bunch of getter functions
  /*This function calculates the rmse iff true test densities are known */
  double get_rmse(){
    
    index_t len=true_test_densities_.n_cols();
    
    if(len==0){
      
      return -1.0;
    }
    
    double diff=0;
    double total_sqd_diff=0;
    for(index_t i=0;i<len;i++){
      
      diff=test_densities_[i]-true_test_densities_.get(0,i);
      total_sqd_diff+=diff*diff;
    }
    return sqrt(total_sqd_diff/num_test_points_); 
  }

  void get_test_densities(Vector *result){

    printf("Test densities are..\n");
    test_densities_.PrintDebug();
    printf("Initialized with %d points",num_test_points_);
    result->Init(num_test_points_);
   
    result->CopyValues(test_densities_);
  }

 private:
  
  void FindBandwidthOfTestPoints_(){


    //We have a separate train and test datasets
    
    //Before u call AllkNN set vertain parameters of the module

    fx_set_param_int(module_,"leaf_size", LEAF_SIZE);

    fx_set_param_int(module_,"knns", num_of_neighbours_);
    AllkNN all_knn;
    ArrayList<index_t> resulting_neighbors;
    ArrayList<double> squared_distances; 
       

    all_knn.Init(test_data_,train_data_,module_);
    printf("Initialized allknn..\n");
    all_knn.ComputeNeighbors(&resulting_neighbors, &squared_distances);
    printf("Computed neighbours of all points..\n");
    
    printf("length of squared distances is %d..\n",squared_distances.size());
    for(index_t i = 0; i < squared_distances.size(); i += num_of_neighbours_) 
      {
	
	
	bandwidth_test_points_[i / num_of_neighbours_]=
	  sqrt(squared_distances[i+num_of_neighbours_-1]);
      
      }

    printf("bandwidth of test points is ..\n");

    bandwidth_test_points_.PrintDebug();  
  }


  void PerformNaiveKDE_(){
    
    for(index_t q = 0; q <num_test_points_;q++) {
      
      //Get the test point
      const double *q_col = test_data_.GetColumnPtr(q);
      
      //First initialize the kernel with the bandwidth of the test
      //point

      gk_.Init(bandwidth_test_points_[q]);
      
      // Compute unnormalized sum first.
      for(index_t r = 0; r < train_data_.n_cols(); r++) {
	
	//Get the reference point
	const double *r_col = train_data_.GetColumnPtr(r);
	double dsqd = 
	  la::DistanceSqEuclidean(train_data_.n_rows(), q_col, r_col);

	test_densities_[q] += gk_.EvalUnnormOnSq(dsqd);
      }

      //Calculate the norm constant

      double norm_const=gk_.CalcNormConstant(num_dims_);
     
      if(norm_const<pow(10,-6)){

	printf("Bandiwdth is 0000000000..\n");
	printf("bandiwdth is %f..\n",bandwidth_test_points_[q]);
      } 
      // Then normalize it.
      test_densities_[q] /= ((num_train_points_)*norm_const);
    }
    fx_timer_stop(module_, "naive_kde_compute");
  }
  
  
 public:


  void ComputeTestDensities(){

    //First find the bandwidth of the test point
    
    FindBandwidthOfTestPoints_();

    printf("Found bandwidths of test points....\n");
    PerformNaiveKDE_();

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
    test_densities_.SetZero();

    //Initialize the bandwidths of all test points

    bandwidth_test_points_.Init(num_test_points_);

    //The number of neighbours
    
    num_of_neighbours_=
      fx_param_int_req(module_,"k");

    printf("Number of neighbours is %d..\n",num_of_neighbours_);
    printf("Succesfully initializesd...\n");

    printf("Number of test points is %d..\n",test_data_.n_cols());
    printf("Number of train points is %d..\n",train_data_.n_cols());
  }
 

};

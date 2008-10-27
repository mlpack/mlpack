#include "interior_point_pred_corr.h"
#include "special_la.h"
#include "fastlib/fastlib_int.h"
#include "ichol.h"
#include "mlpack/kde/bandwidth_lscv.h"
#include "mlpack/kde/naive_kde.h"
#include "mlpack/series_expansion/kernel_aux.h"


int main(int argc, char *argv[]){
  
  //Initalize FastExec
  fx_init(argc,argv,NULL);
  
  struct datanode* ipc=fx_submodule(NULL,"ipc");
  
  //Train data file is a reuqirement
  const char *train_file=fx_param_str_req(ipc,"train");


  index_t num_train_points;  
  Matrix train_data;
  
  //Load the train datasets
  
  data::Load(train_file,&train_data);
  num_train_points=train_data.n_cols();

  printf("Train dataset has been read....\n");

  index_t num_test_points=0;

  //This is a flag
  index_t test_file_present=0;

  Matrix test_data;
  Vector computed_test_densities;
  //Check for the existence of a query set and accordingly read data
  if(fx_param_exists(ipc,"test")){
    
    //Load the dataset
    const char *test_file=fx_param_str_req(ipc,"test");
    data::Load(test_file,&test_data);
    test_file_present=1;
    num_test_points=test_data.n_cols();
  }
  else{
    
    test_data.Init(0,0); //This avoids segmentation fault
    num_test_points=0;
    computed_test_densities.Init(num_test_points);
  }
  
  //Having got all the initial data lets create an object of the class
  //HkInteriorPointPredictorCorrector
  
  HKInteriorPointPredictorCorrector hk_ippc; 
  hk_ippc.Init(train_data,test_data,ipc);
  hk_ippc.ComputeOptimalSolution();


  Vector hk_test_densities;
  //Get the test densities
  if(test_file_present==1){

    //Dont need to initialize the vector hk_test_densities
    hk_ippc.get_test_densities(hk_test_densities);
  }
  else{

    //This prevents segmentation fault
    hk_test_densities.Init(0);
  }

  double neg_log_likeli_test_hyperkernel=
    hk_ippc.get_negative_log_likelihood_test();

  double rmse_hyperkernel_test= hk_ippc.get_rmse_test();
  
  /////////////// NAIVE KDE AND CROSSVALIDATION////////////
    struct datanode* naive=fx_submodule(NULL,"naive");
 
  //Get the minimum bandwidth for naive kde using least squares
  //crossvalidation
  double min_band_naive_kde;
  
  /*Matrix reference_weights;
  reference_weights.Init(num_train_points,1);
  reference_weights.SetAll(1);
  min_band_naive_kde=BandwidthLSCV::
    Optimize<GaussianKernelAux>(train_data,reference_weights);
  
  Vector naive_kde_test_densities;

  double negative_log_likelihood_test_naive=1;
  if(fx_param_exists(NULL,"test")){
    
    //Uniform weight KDE
    fx_set_param_double(naive,"bandwidth", min_band_naive_kde);
    naive_kde.Init(train_data,test_data,naive);
    naive_kde.Compute();
    naive_kde.get_density_estimates(&naive_kde_test_densities);
   
    for(index_t i=0;i<num_test_points;i++){
      
      negative_log_likelihood_test_naive*=naive_kde_test_densities[i];
    }

    negative_log_likelihood_test_naive=
      -log(negative_log_likelihood_test_naive);
  }
  else{

    negative_log_likelihood_test_naive=DBL_MIN;
  }

  negative_log_likelihood_test_naive=-log(negative_log_likelihood_test_naive);
  
  //Having got the density estimates calculate the mean squared error
  //Only if the true densities are known

  if(fx_param_exists(module_,"true")){

    data::Load(true_density_file,&true_test_densities_);
    
    double rmse=0;
    for(index_t i=0;i<num_test_points_;i++){
      
      double diff=
	naive_kde_test_densities_[i]-true_test_densities_.get(0,i);
      rmse=diff*diff;
    }
    }*/
  
  fx_done(NULL);
}

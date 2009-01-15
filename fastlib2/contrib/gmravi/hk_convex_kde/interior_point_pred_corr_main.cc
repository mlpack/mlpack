/*
  This code will simply run the hyperkernel density estimation code
  with parameter settings


*/


#include "interior_point_pred_corr2.h"
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


  //Variables concerning test set

  index_t num_test_points=0;
  index_t test_file_present=0;
  index_t true_density_file_present=0;
  Matrix test_data;

  //Check for the existence of a test set and accordingly read data
  if(fx_param_exists(ipc,"test")){
    
    //Load the dataset
    const char *test_file=fx_param_str_req(ipc,"test");
    data::Load(test_file,&test_data);
    test_file_present=1;
    num_test_points=test_data.n_cols();
  }
  else{
    printf("Test set doesnt exist.....\n");
    
    test_data.Init(0,0); //This avoids segmentation fault
    num_test_points=0;
  }


  Matrix true_test_densities;
  if(fx_param_exists(ipc,"true")){
    true_density_file_present=1;
    
    const char *true_density_file=fx_param_str_req(ipc,"true");
    //Since the true test densities are given, hence
    data::Load(true_density_file,&true_test_densities);
    
  }


  /**************************** Do naive kde first*******************************/    

  //Lets do least squares cv for naive kde

  /////////////// NAIVE KDE AND CROSSVALIDATION////////////
  
  //Get the minimum bandwidth for naive kde using least squares
  //crossvalidation

  struct datanode* naive=fx_submodule(NULL,"naive");
  double min_band_naive_kde;
  
  Matrix reference_weights;
  reference_weights.Init(1,num_train_points);
  reference_weights.SetAll(1);

  printf("Reference weights are all set...\n");

  printf("Will optimize for least squares bandwidth...\n");
  min_band_naive_kde=
    BandwidthLSCV::
    Optimize<GaussianKernelAux>(train_data,reference_weights);

  printf("Finished crossvalidation for least squares bandwidth");
  

  printf("Minimum bandwidth is %f\n",min_band_naive_kde);
  
  double optimal_ise_score_naive_kde=
    BandwidthLSCV::
    ComputeLSCVScore<GaussianKernelAux>(train_data,reference_weights,min_band_naive_kde);
  
    printf("Least squares crossvalidation score at the optimal bandwidth for naive kde is %f..\n",optimal_ise_score_naive_kde);
  
    Vector naive_kde_test_densities;
    
    
    //Uniform weight KDE
    
    // NaiveKde<GaussianKernel> naive_kde;
    //   fx_set_param_double(naive,"bandwidth", min_band_naive_kde);
    //   fx_set_param_str(naive,"mode","fixedbw");
    
    //  naive_kde.Init(test_data,train_data,naive);
    //      naive_kde.Compute();
    //      naive_kde.get_density_estimates(&naive_kde_test_densities);
    
    GaussianKernel gk;
    gk.Init(min_band_naive_kde);

    index_t num_dims=train_data.n_rows();
    
    naive_kde_test_densities.Init(num_test_points);
    double norm_const=
      gk.CalcNormConstant(num_dims);
    
    for(index_t i=0;i<num_test_points;i++){
      
      double *x=test_data.GetColumnPtr(i);
      
      double total_contrib=0;
      
      for(index_t j=0;j<num_train_points;j++){
	
	double *x_j=train_data.GetColumnPtr(j); 
	
	double sqd_dist=
	  la::DistanceSqEuclidean(num_dims,x, x_j);
	
	double unnorm_kernel_val=
	  gk.EvalUnnormOnSq(sqd_dist);
	
	total_contrib+=unnorm_kernel_val;
      }
      naive_kde_test_densities[i]=total_contrib/(num_train_points*norm_const);
    }
    
   //Having got the density estimates calculate the mean squared error
   //Only if the true densities are known
   

   double rmse_naive_kde_opt=-1;

   
   if(fx_param_exists(ipc,"true")){
     
     double rmse=0;
     index_t num_test_points=true_test_densities.n_cols();
     
     for(index_t i=0;i<num_test_points;i++){
       
       double diff=
	 naive_kde_test_densities[i]-true_test_densities.get(0,i);
       
       rmse+=diff*diff;
     }
     rmse_naive_kde_opt=sqrt(rmse/num_test_points);
     true_density_file_present=1;

     printf("Optimal bw for naive kde is %f..\n",min_band_naive_kde);
     printf("RMSE naive kde is %f..\n",rmse_naive_kde_opt);
   }
   else{
     
     true_test_densities.Init(0,0);
   }

   //You can always calculate the likelihood

   // double negative_log_likelihood_naive_kde_opt=1;

//    for(index_t i=0;i<num_test_points;i++){

//      negative_log_likelihood_naive_kde_opt*=naive_kde_test_densities[i];
//    }

//    negative_log_likelihood_naive_kde_opt=-log(negative_log_likelihood_naive_kde_opt);

   /****************************************Done with naive kde**********************************/





   fx_timer_start(NULL,"full");


   //Number of parameteres for crossvalidation

   index_t num_sigma=5;
   index_t num_sigma_h=5;
   index_t num_lambda=1;


   Vector sigma_vec;
   Vector sigma_h_vec;
   Vector lambda_vec;

   //Initialize these vectors

   sigma_vec.Init(num_sigma+1);
   sigma_h_vec.Init(num_sigma_h+1);
   lambda_vec.Init(num_lambda);


   //Set up these values
   double min_sigma=min_band_naive_kde/3;
   double max_sigma=3*min_band_naive_kde;

   double gap=(max_sigma-min_sigma)/num_sigma;

   for(index_t i=0;i<num_sigma;i++){
     
     sigma_vec[i]=min_sigma+i*gap;
   }
   sigma_vec[num_sigma]=min_band_naive_kde/sqrt(2);


   //Set up sigma_h

   gap=min_sigma/(2*num_sigma_h);

   double average_sigma=(min_sigma+max_sigma)/2.0;
   for(index_t i=0;i<num_sigma_h;i++){

     sigma_h_vec[i]=i*average_sigma;     
   } 
   sigma_h_vec[num_sigma_h]=60; //Infinite sigma_h

   lambda_vec[0]=200.0;
  
  //Having got all the initial data lets create an object of the class
  //HkInteriorPointPredictorCorrector

  double min_ise_hkde=DBL_MAX;
 

  printf("Sigma vector is ...\n");
  sigma_vec.PrintDebug();

  printf("Sigma_h vector is ...\n");
  sigma_h_vec.PrintDebug();
  
  printf("lambda vector is..\n");
  lambda_vec.PrintDebug();

  /*
    A little hack 

  */

//   num_sigma=0;
//   num_sigma_h=0;
//   num_lambda=1;
  
//   sigma_vec[0]=min_band_naive_kde/sqrt(2);
//   sigma_h_vec[0]=60;
//   lambda_vec[0]=10;
  


 
  double reg_ise_opt=DBL_MAX; 
  index_t num_defaulters=0;      

  double sigma_opt,sigma_h_opt,lambda_opt;

  for(index_t i=0;i<num_sigma+1;i++){
    
    for(index_t j=0;j<num_sigma_h+1;j++){
      
      for(index_t k=0;k<num_lambda;k++){


	fx_set_param_double(ipc,"sigma",sigma_vec[i]);
	fx_set_param_double(ipc,"sigma_h",sigma_h_vec[j]);      
	fx_set_param_double(ipc,"lambda",lambda_vec[k]);
	
	
	HKInteriorPointPredictorCorrector hk_ippc; 
	hk_ippc.Init(train_data,test_data,ipc);
	index_t ret_val=hk_ippc.ComputeOptimalSolution();
	
	if(ret_val==-1){
	  
	  //do nothing
	  num_defaulters++;
	}
	else{
	  double reg_ise=
	    hk_ippc.get_penalized_ise_train_set();
	  
	  if(reg_ise<reg_ise_opt){
	    
	    sigma_opt=sigma_vec[i];
	    sigma_h_opt=sigma_h_vec[j];
	    lambda_opt=lambda_vec[k];
	    reg_ise_opt=reg_ise;
	  }
	}
      }
    }
  }

  //Fianlly run with the optimal settings
  
  fx_set_param_double(ipc,"sigma",sigma_opt);
  fx_set_param_double(ipc,"sigma_h",sigma_h_opt);      
  fx_set_param_double(ipc,"lambda",lambda_opt);
  
  
  HKInteriorPointPredictorCorrector hk_ippc; 
  hk_ippc.Init(train_data,test_data,ipc);
  hk_ippc.ComputeOptimalSolution();
  double ise_opt_train_set_hkde=
    hk_ippc.get_ise_train_set();

  Vector test_densities_hkde_opt;
  hk_ippc.get_test_densities(test_densities_hkde_opt);
  
  printf("The number of defaulters are %d..\n",num_defaulters);

  /////////////////////////////////////////////////////////////////////////////////////
  
 //  printf("Run the code with optimal hyperkernel parameters....\n");
  
  //   printf("sigma_h=%f,sigma=%f,lambda=%f..\n",opt_sigma_h,opt_sigma,opt_lambda);
  
  //   ///////Run HKDE WITH OPTIMAL PARAMETERS
  //   HKInteriorPointPredictorCorrector hk_ippc; 
  //   fx_set_param_double(ipc,"sigma_h",opt_sigma_h);
  //   fx_set_param_double(ipc,"sigma",opt_sigma);
  //   fx_set_param_double(ipc,"lambda",opt_lambda);
  
  //   hk_ippc.Init(train_data,test_data,ipc);
  //   hk_ippc.ComputeOptimalSolution();
  //   double hkde_ise_opt=hk_ippc.get_ise_train_set();
  //   double negative_log_likelihood_hkde_opt=
  //     hk_ippc.get_negative_log_likelihood_test();
  
  //   Vector test_densities_hkde_opt;
  //   hk_ippc.get_test_densities(test_densities_hkde_opt);
  
  //////////////////////////////////////////////////////////////////////////
  

  //Calculate RMSE of hkde. To do this check to see if we have a
  //true_density_file
 
  double rmse_hkde_opt=-1;
  
  if(true_density_file_present==1){
    
    double rmse=0;
    for(index_t i=0;i<num_test_points;i++){

      //printf("Actual density is %f..\n",true_test_densities.get(0,i));
      //printf("Calculated density is %f..\n",test_densities_hkde_opt[i]);
      
      double diff=
	test_densities_hkde_opt[i]-true_test_densities.get(0,i);
      
      rmse+=diff*diff;
    }
    rmse_hkde_opt=sqrt(rmse/num_test_points);
  }

  fx_timer_stop(NULL,"full");

  //print all the results  
  printf("HYPERKERNEL results...................................\n");
  printf("sigma_h=%f\n",sigma_h_opt);
  printf("sigma=%f\n",sigma_opt);
  printf("lambda=%f\n",lambda_opt);
  printf("The unregualrized ISE of train set is %f..\n",
	 ise_opt_train_set_hkde);
  printf("The regualrized ISE of train set is %f..\n",reg_ise_opt);
  
  
  printf("HKDE Densities are............\n");
  test_densities_hkde_opt.PrintDebug();
  

  //printf("penalized ise at optimal settings is %f..\n",min_penalized_ise_hkde);
  printf("RMSE:%f....\n",rmse_hkde_opt);

  printf("..............................\n");
  printf("NAIVE KDE results...\n");

  printf("The bandwidth of naive kde is %f..\n",min_band_naive_kde);
  printf("The ise of naive kde at optimal setting is %f..\n",optimal_ise_score_naive_kde);
  printf("rmse of naive kde is %f..\n",rmse_naive_kde_opt);
  printf("Naive kde densities are...\n");
  naive_kde_test_densities.PrintDebug();


  ////////////Print Them//////////////////
  
  // FILE *fp_naive=fopen("naive_kde_densities_mog9.txt","w");
  //    FILE *fp_hkde=fopen("hkde_densities_mog9.txt","w");
  
  //    test_densities_hkde_opt.PrintDebug(NULL,fp_hkde);
  //    naive_kde_test_densities.PrintDebug(NULL,fp_naive);
  
  fx_done(NULL);
}

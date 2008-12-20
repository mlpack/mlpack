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

  index_t num_test_points=0;

  //This is a flag
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
    
    test_data.Init(0,0); //This avoids segmentation fault
    num_test_points=0;
  }
  

  //Lets do least squares cv for naive kde


  /////////////// NAIVE KDE AND CROSSVALIDATION////////////
  struct datanode* naive=fx_submodule(NULL,"naive");  
  
  //Get the minimum bandwidth for naive kde using least squares
  //crossvalidation
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
    BandwidthLSCV::ComputeLSCVScore<GaussianKernelAux>(train_data,reference_weights,
						       min_band_naive_kde);
 

  printf("Least squares crossvalidation score at the optimal bandwidth is %f..\n",optimal_ise_score_naive_kde);


  Vector naive_kde_test_densities;

   if(fx_param_exists(ipc,"test")){
    
    //Uniform weight KDE

     NaiveKde<GaussianKernel> naive_kde;
     fx_set_param_double(naive,"bandwidth", min_band_naive_kde);
     fx_set_param_str(naive,"mode","fixedbw");
     
     naive_kde.Init(test_data,train_data,naive);
     naive_kde.Compute();
     naive_kde.get_density_estimates(&naive_kde_test_densities);

     printf("Test points for naive kde are..\n");
     test_data.PrintDebug();
     printf("Naive kde test densities are...\n");
     naive_kde_test_densities.PrintDebug();
   }
   else{
     
     naive_kde_test_densities.Init(0);
   }


   //Having got the density estimates calculate the mean squared error
   //Only if the true densities are known
   

   double rmse_naive_kde_opt=DBL_MAX;



   Matrix true_test_densities;
   if(fx_param_exists(ipc,"true")){
     
     const char *true_density_file=fx_param_str_req(ipc,"true");
     //Since the true test densities are given, hence
     data::Load(true_density_file,&true_test_densities);
     double rmse=0;
     index_t num_test_points=true_test_densities.n_cols();
     
     for(index_t i=0;i<num_test_points;i++){
       
       double diff=
	 naive_kde_test_densities[i]-true_test_densities.get(0,i);
       
       rmse+=diff*diff;
     }
     rmse_naive_kde_opt=sqrt(rmse/num_test_points);
     true_density_file_present=1;

     printf("RMSE naive kde is %f..\n",rmse_naive_kde_opt);
   }
   else{
     
     true_test_densities.Init(0,0);
   }

   //You can always calculate the likelihood

   double negative_log_likelihood_naive_kde_opt=1;

   for(index_t i=0;i<num_test_points;i++){

     negative_log_likelihood_naive_kde_opt*=naive_kde_test_densities[i];
   }

   negative_log_likelihood_naive_kde_opt=-log(negative_log_likelihood_naive_kde_opt);

  

  ////////////////////////////////////////////////////////////////////////////////

   fx_timer_start(NULL,"full");
   double min_sigma=min_band_naive_kde*0.50;
   
   double max_sigma=min_band_naive_kde*1.25;

   index_t number_steps=5;
   
   double gap=(max_sigma-min_sigma)/number_steps;

   Vector sigma;
   sigma.Init(number_steps+1);


   //TODO: CHANGE THIS CHANGE THIS

   min_sigma=min_band_naive_kde;
   
   for(index_t i=0;i<number_steps;i++){
     
     sigma[i]=min_sigma+(i*gap);
   }
   
   sigma[number_steps]=min_band_naive_kde/sqrt(2);

   printf("Sigma is..\n");
   sigma.PrintDebug();
  
   Vector lambda;
   
   lambda.Init(1);
   lambda[0]=0.1;
   // lambda[1]=0.5;
   //lambda[2]=0.05;
   //lambda[3]=5;
   //lambda[4]=20;

  //Having got all the initial data lets create an object of the class
  //HkInteriorPointPredictorCorrector

  //Decide upon a range for \sigma,\sigma_h and lambda
  
  //************************************************



  double min_ise_hkde=DBL_MAX;
  double opt_sigma_h;
  double opt_sigma;
  double opt_lambda;


  index_t total_crossvalidation=0;
  index_t num_sigma_h=5;

  Vector sigma_h;
  sigma_h.Init(num_sigma_h+1);

  Matrix dummy_test;
  dummy_test.Init(0,0);


  double ise_at_inf_sigma_h=DBL_MAX;

  for(index_t j=0;j<sigma.length();j++){

    fx_set_param_double(ipc,"sigma",sigma[j]);

    double min_sigma_h=0;
    double max_sigma_h=4*sigma[j];
    
    printf("sigma_h is..\n");
    sigma_h.PrintDebug();
    double gap=(max_sigma_h-min_sigma_h)/num_sigma_h;
    for(index_t p=0;p<num_sigma_h;p++){
      
      sigma_h[p]=min_sigma_h+(p*gap);
      printf("sigma_h has added in %f..\n",sigma_h[p]);
    }
    
    sigma_h[num_sigma_h]=60; //inifinte sigma_h
     printf("Sigma_h is...\n");
     sigma_h.PrintDebug();
    
    for(index_t i=0;i<sigma_h.length();i++){
      
      fx_set_param_double(ipc,"sigma_h",sigma_h[i]);      
      
      for(index_t k=0;k<lambda.length();k++){
	
	fx_set_param_double(ipc,"lambda",lambda[k]);
	

	printf("TOTAL ITERATIONS DONE ARE %d...\n",total_crossvalidation);
	printf("sigma_h=%f,sigma=%f,lambda=%f...\n",sigma_h[i],sigma[j],lambda[k]);

	HKInteriorPointPredictorCorrector hk_ippc; 

	hk_ippc.Init(train_data,dummy_test,ipc);
	index_t ret_val=hk_ippc.ComputeOptimalSolution();

	double ise;
	if(ret_val==-1){
	  printf("There was a possibiltity of inf loop......\n");
	  printf("sigma_h=%f,sigma=%f,lambda=%f...\n",sigma_h[i],sigma[j],lambda[k]);
	  ise=DBL_MAX;
	}
	else{
	  ise=hk_ippc.get_ise_train_set();
	}
	if(ise<min_ise_hkde){

	  min_ise_hkde=ise;
	  opt_sigma_h=sigma_h[i];
	  opt_sigma=sigma[j];
	  opt_lambda=lambda[k];
	}

	if(fabs(sigma_h[i]-60)<0.000001&&fabs(sigma[j]-min_band_naive_kde/sqrt(2))<0.000001){

	  ise_at_inf_sigma_h=ise;
	}
	total_crossvalidation++;
      }
    }
  }
  /////////////////////////////////////////////////////////////////////////////////////
  
  printf("Run the code with optimal hyperkernel parameters....\n");
  
  printf("sigma_h=%f,sigma=%f,lambda=%f..\n",opt_sigma_h,opt_sigma,opt_lambda);

  ///////Run HKDE WITH OPTIMAL PARAMETERS
  HKInteriorPointPredictorCorrector hk_ippc; 
  fx_set_param_double(ipc,"sigma_h",opt_sigma_h);
  fx_set_param_double(ipc,"sigma",opt_sigma);
  fx_set_param_double(ipc,"lambda",opt_lambda);
  
  hk_ippc.Init(train_data,test_data,ipc);
  hk_ippc.ComputeOptimalSolution();
  double hkde_ise_opt=hk_ippc.get_ise_train_set();
  double negative_log_likelihood_hkde_opt=
    hk_ippc.get_negative_log_likelihood_test();

  Vector test_densities_hkde_opt;
  hk_ippc.get_test_densities(test_densities_hkde_opt);

  //////////////////////////////////////////////////////////////////////////

  double rmse_hkde_opt=DBL_MAX;

  if(true_density_file_present==1){

    double rmse=0;
    for(index_t i=0;i<num_test_points;i++){
      
      double diff=
	test_densities_hkde_opt[i]-true_test_densities.get(0,i);
      
      rmse+=diff*diff;
    }
    rmse_hkde_opt=sqrt(rmse/num_test_points);
  }

  fx_timer_stop(NULL,"full");

  //print all the results  
  printf("HYPERKERNEL results...................................\n");
  printf("sigma_h_opt=%f\n",opt_sigma_h);
  printf("sigma_opt=%f\n",opt_sigma);
  printf("lambda_opt=%f\n",opt_lambda);
  printf("Negative lOG LIKELIHOOD=%f...\n",negative_log_likelihood_hkde_opt);
  printf("ISE:%f\n",min_ise_hkde);
  //printf("penalized ise at optimal settings is %f..\n",min_penalized_ise_hkde);
  printf("RMSE:%f....\n",rmse_hkde_opt);
  printf("ise at infinite sigma_h=%f..\n",ise_at_inf_sigma_h);

  printf("............................\n");

  printf("NAIVE KDE results...\n");

  printf("h_opt:%f\n...",min_band_naive_kde);
  printf("NEGATIVE LOG LIKELIHOOD=%f..\n",negative_log_likelihood_naive_kde_opt);
  printf("ISE TRAIN=%f..\n",optimal_ise_score_naive_kde);
  printf("RMSE=%f.........\n",rmse_naive_kde_opt);

  printf("Total number of iterations are %d...\n",total_crossvalidation);




  printf("HKDE Densities are...\n");
   test_densities_hkde_opt.PrintDebug();


  printf("NAIVE KDE Densities are...\n");
  naive_kde_test_densities.PrintDebug();

  ////////////Print Them//////////////////
  
   FILE *fp_naive=fopen("naive_kde_densities_mog9.txt","w");
   FILE *fp_hkde=fopen("hkde_densities_mog9.txt","w");
  
   test_densities_hkde_opt.PrintDebug(NULL,fp_hkde);
   naive_kde_test_densities.PrintDebug(NULL,fp_naive);
  
   fx_done(NULL);
}

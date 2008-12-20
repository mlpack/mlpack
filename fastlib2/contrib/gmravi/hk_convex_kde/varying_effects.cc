#include "interior_point_pred_corr2.h"
#include "special_la.h"
#include "fastlib/fastlib_int.h"
#include "ichol.h"
#include "mlpack/kde/bandwidth_lscv.h"
#include "mlpack/kde/naive_kde.h"
#include "mlpack/series_expansion/kernel_aux.h"

double get_rmse(Vector &test_densities,Matrix &true_densities){

  if(true_densities.n_cols()==0){

    return DBL_MAX;
  }
  
  index_t num_test_points=true_densities.n_cols();
  double rmse=0;
  for(index_t i=0;i<num_test_points;i++){
    
    double diff=
      test_densities[i]-true_densities.get(0,i);
    
    rmse+=diff*diff;
  }
  rmse=sqrt(rmse/num_test_points);
  return rmse;
}


////THE SOLE PURPOSE OF THIS FILE IS TO STUDY THE EFFECTS OF VARYING DIFF PARAMETERS

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

  ///////////////////////////////////////////////////////////////////////////////////



  Matrix true_densities;
  //  Read the true densities file
  if(fx_param_exists(ipc,"true")){

    const char *true_file=fx_param_str_req(ipc,"true");
    data::Load(true_file,&true_densities);
  }
  else{
    
    true_densities.Init(0,0);
  }

  ArrayList<double> rmse_hkde;
  ArrayList<double> ise_hkde;

  rmse_hkde.Init(0);
  ise_hkde.Init(0);

  fx_timer_start(NULL,"full");






  //////////SET SIGMA//////////////////////
  double min_sigma=min_band_naive_kde/2;
  
  double max_sigma=min_band_naive_kde*1.25;
  
  index_t num_sigma=0;
  
  double gap=(max_sigma-min_sigma)/num_sigma;
  
  Vector sigma;
  sigma.Init(num_sigma+1);
  
 
  
  for(index_t i=0;i<num_sigma;i++){
    
    sigma[i]=min_sigma+(i*gap);
  }
  
  sigma[num_sigma]=min_band_naive_kde/sqrt(2);
  
  printf("Sigma is..\n");
  sigma.PrintDebug();
  


  //////////////SET LAMBDA/////////////////////
  Vector lambda;
  
  index_t num_lambda=8;
  lambda.Init(num_lambda);
  lambda[0]=0.1;
  lambda[1]=0.001;
  lambda[2]=0.3;
  lambda[3]=0.5;
  lambda[4]=0.7;
  lambda[5]=0.9;
  lambda[6]=2;
  lambda[7]=10;
  ////////////////////////////////////////

  printf("lambda is...\n");
  lambda.PrintDebug();


  //Having got all the initial data lets create an object of the class
  //HkInteriorPointPredictorCorrector
  
  //Decide upon a range for \sigma,\sigma_h and lambda
  
  //************************************************
 
 
  index_t total_crossvalidation=0;


  ///////////SET SIGAM_H/////////////////
  index_t num_sigma_h=0;
  
  Vector sigma_h;
  sigma_h.Init(num_sigma_h+1);
 
  for(index_t j=0;j<sigma.length();j++){
    
    fx_set_param_double(ipc,"sigma",sigma[j]);

    double min_sigma_h=0;
    double max_sigma_h=5*sigma[j];
    printf("min sigma_h is...%f\n",min_sigma_h);

    double gap=(max_sigma_h-min_sigma_h)/num_sigma_h;
    for(index_t p=0;p<num_sigma_h;p++){
      
      sigma_h[p]=min_sigma_h+(p*gap);
      
      printf("sigma_h has added in %f..\n",sigma_h[p]);
    }
    sigma_h[num_sigma_h]=0; //infinite sigma_h

    printf("Sigma_h is...\n");
    sigma_h.PrintDebug();
    
    for(index_t i=0;i<sigma_h.length();i++){
      
      fx_set_param_double(ipc,"sigma_h",sigma_h[i]);      
      
      for(index_t k=0;k<lambda.length();k++){
	
	fx_set_param_double(ipc,"lambda",lambda[k]);
	

	printf("TOTAL ITERATIONS DONE ARE %d...\n",total_crossvalidation);
	

	HKInteriorPointPredictorCorrector hk_ippc; 

	hk_ippc.Init(train_data,test_data,ipc);
	index_t ret_val=hk_ippc.ComputeOptimalSolution();
	printf("ret_value is %d..\n",ret_val);

	if(ret_val!=-1){

	  double ise=hk_ippc.get_ise_train_set();

	  ise_hkde.AddBackItem(ise);

	  
	  Vector test_densities_hkde;
	  hk_ippc.get_test_densities(test_densities_hkde);

	  double rmse=get_rmse(test_densities_hkde,true_densities);
	  rmse_hkde.AddBackItem(rmse);
	  //printf("lambda=%f,ise=%f and rmse=%f..\n",lambda[k],ise,rmse);
	}
	total_crossvalidation++;
      }
    }
  }
  /////////////////////////////////////////////////////////////////////////////////////
  
  fx_timer_stop(NULL,"full");

  //print all the results  


  printf("//////////////////////////////////////////////////////////////////////////////////////");
  printf("ise of naive kde is %f..\n",optimal_ise_score_naive_kde);
  printf("Optimal naive kde bandwidth =%f..\n",min_band_naive_kde);



  printf("Printing results...\n");

  printf("NUmber of train points are %d..\n",num_train_points);

  printf("sigma=%f,sigma_h=%f\n",sigma[0],sigma_h[0]);

  printf("size of rmse_hkde is %d",rmse_hkde.size());

  for(index_t i=0;i<rmse_hkde.size();i++){

    printf("lambda=%f,rmse=%f,ise=%f..\n",lambda[i],rmse_hkde[i],ise_hkde[i]);
  }

  printf("lambda is ..\n");
  lambda.PrintDebug();

  
  // FILE *fp_naive=fopen("naive_kde_densities_mog9.txt","w");
  //FILE *fp_hkde=fopen("hkde_densities_mog9.txt","w");
  
  //test_densities_hkde_opt.PrintDebug(NULL,fp_hkde);
  //naive_kde_test_densities.PrintDebug(NULL,fp_naive);
  
   fx_done(NULL);
}



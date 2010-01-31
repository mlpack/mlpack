#include "fastlib/fastlib.h"
#include "dataset_scaler.h"
#include "naive_kde_local_polynomial.h"
#include "vector_kernel.h"
#include "naive_kde.h"
#include "naive_local_likelihood.h"



int main(int argc, char *argv[]){

  fx_init(argc,argv);


  printf("Entered the function.....\n");

  ///Reading parameters

  struct datanode * nklpm=
    fx_submodule(NULL,"nklpm","nklpm");
  
  
  //The reference file name
  const char *ref_file_name=fx_param_str_req(nklpm,"data");
  
  //The query points
  const char *query_file_name=fx_param_str_req(nklpm,"query");
  
  //The bandwidth file
  const char *bandwidth_file=fx_param_str(nklpm,"bandwidth","default");

  //The true density file name

  const char *true_density_file_name=
    fx_param_str(nklpm,"true_density","default");


  bool LOCAL_LIKELIHOOD=false; // A flag to be set if local likelihood
			 // calculations are required

  bool NAIVE_KDE=false; //A flag to be set if naive kde calculations
			//are required

  bool TRUE_DENSITY_GIVEN=false; // A flag to be set if the true
				 // density file is given

  Matrix references;
  Matrix query;
  Matrix  bandwidths;  

  Vector naive_kde_lp_results;
  Vector naive_kde_results;
  Vector naive_ll_results;

  Matrix  true_density;
 
  //Load the reference file
  data::Load(ref_file_name,&references);
  
  //Load the query file
  data::Load(query_file_name,&query);



  /////////////////////////READ FILES///////////////////////

  Vector boundary_points;
  boundary_points.Init(3*references.n_rows());
  boundary_points.SetAll(DBL_MIN);
  
 
  //If true density file is provided then get the boundary points

 if(strcmp(true_density_file_name,"default")){

    if(references.n_rows()==1){
   
      //In case the true density has been provided then compare the
      //local polynomial density estimates with the true density
      
      DatasetScaler::GetBoundaryPoints(references,boundary_points);
    }
  }
  
  
  //Load the bandwidths
  if(strcmp(bandwidth_file,"default")){

    //BW file has been provided
    printf("Bandwidth file has been provided...\n");
    data::Load(bandwidth_file,&bandwidths);
  }
  else{
    //NO B.W file. 1-D=>plugin bandwidth
    // >= 2-D => arbitrary bandwdith setting

    bandwidths.Init(1,query.n_rows());
    
  
    
    if(references.n_rows()==1){

      //Set -ve b.w. Will find plugin b.w later
      bandwidths.SetAll(-1);
      printf("Set all bandwidths to negative...\n");
    }
    else{
      
      //multi-dimensional case. Set all the bandiwdths to 0.20
      bandwidths.SetAll(0.2);
    } 
  }

  //For numerical stability scale the dataset to a unit cube
  if(!strcmp(fx_param_str(nklpm,"scale","false"),"true")){

    DatasetScaler::ScaleDataByMinMax(query, references, false);
  }
  

  //Load the true density file
  if(strcmp(true_density_file_name,"default")){

    //Since true desnity has been provided
    TRUE_DENSITY_GIVEN=true;

    

    //This means that the user has provided a true density  file
    
    data::Load(true_density_file_name,&true_density);
  }
  else{
    //Since no true density file has not been provided, hence we shall
    //use 0 for all values
    
    
    true_density.Init(1,query.n_rows());
    
    //This is being done so as to communicate
    //to the program that it needs to do plugin
    //bandwidth calculations
    
    true_density.SetAll(0);
  }


  ////////////////////////////////////////////////////////////////

  /////////////////LOCAL POLYNOMIAL KDE CALCULATIONS//////////////////

  NaiveKdeLP <ErfDiffKernel> naive_kde_lp;
  naive_kde_lp.Init(query,references,
		    bandwidths.GetColumnPtr(1),
		    bandwidths.n_rows());
  naive_kde_lp.Compute();


  //If true density has been given
  if(strcmp(true_density_file_name,"default")){
    naive_kde_lp.Compare(true_density,boundary_points);
  }

  ////////////////////GET RESULTS OF POLYNOMIAL KDE CALCULATIONS///////////

  naive_kde_lp_results.Init(query.n_cols());
  naive_kde_lp.get_density_estimates(naive_kde_lp_results);
  printf("Bandwdiths is ...\n");
  bandwidths.PrintDebug();

  Vector temp; //we shall use this to hold the bandwidth for naive kde calc
  temp.Init(bandwidths.n_rows());

  if(bandwidths.get(0,0)<0){
    
    //This means we used plugin bandwidths for our calculations. Hence
    //lets get the bandwidths used for calculations by using a getter
    //function

    naive_kde_lp.get_bandwidths(temp); 
    printf("Bandwidth used is %f\n",temp[0]);
  }  

  else{

    //this means that the bandwidth file was provided and hence use
    //values from that file
    temp.CopyValues(bandwidths.GetColumnPtr(0));
  }


 
  ////////////////Naive KDE Calculations///////////////

  naive_kde_results.Init(query.n_cols());
  
  if(!strcmp(fx_param_str(nklpm,"naive_kde","false"),"true")){

    //Since Naive Kde  calculations are required set up the flag

    NAIVE_KDE=true;
    
    NaiveKde<GaussianKernel> naive_kde; 
    
    //At the moment naive kde uses just one single bandwidth for all its
    //density calculations. Hence we assume that the bandwidths in all
    //directions are the same
    
    naive_kde.Init(query,references,temp[0]); 
    naive_kde.Compute(&naive_kde_results);

    naive_kde.get_density_estimates(naive_kde_results);
    //Lets compare the results of naive kde calculations with those of
    //naive_kde_lp. This will find the max and min relative erros and
    //will also output the RMSE


    //Check to see if the true density has been provided
     if(strcmp(true_density_file_name,"default")){

       naive_kde.Compare(true_density,boundary_points);
     }
  }

  ////////////Local Likelihood Calculations////////////////////
  
  naive_ll_results.Init(query.n_cols());
  if(!strcmp(fx_param_str(nklpm,"local_likelihood","false"),"true")){

    //Set up the Local likelihood flag

    LOCAL_LIKELIHOOD=true;

    printf("starting local likelihood caclulations...\n"); 

    NaiveLocalLikelihood<GaussianVectorKernel> naive_ll;
 
    naive_ll.Init(query,references,bandwidths.GetColumnPtr(2),
		  bandwidths.n_rows());

    naive_ll.Compute();
    naive_ll.get_local_likelihood_densities(naive_ll_results);

    //Compare the results of naive_ll with naive_kde_lp
    
    if(strcmp(true_density_file_name,"default")){
      
      naive_ll.Compare(true_density,boundary_points);
    }
  }
  
  //Print the results to a file

  if(LOCAL_LIKELIHOOD && NAIVE_KDE && 
     TRUE_DENSITY_GIVEN && 
     references.n_rows()==1){
    
    FILE *fp;
    char *temp;
    temp=(char*)malloc(100*sizeof(char));
    strcpy(temp,"results_");
    printf("True densiy file name is %s\n",true_density_file_name);
    strcat(temp,true_density_file_name);
    fp=fopen(temp,"w+");
    printf("Will be writing file %s\n",temp);

    for(index_t i=0;i<query.n_cols();i++){
      // printf("naive_kde_lp results is %f\n",naive_kde_lp_results[i]);
      fprintf(fp,"%f,%f,%f,%f,%f\n",query.get(0,i),
	      naive_kde_lp_results[i],
	      naive_ll_results[i],
	      naive_kde_results[i],
	      true_density.get(0,i));    
    }
  }

  fx_done();
  return 1; 
}  


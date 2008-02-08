
#include "fastlib/fastlib_int.h"
#include "dataset_scaler.h"
#include "regression_parent.h"


int main(int argc, char *argv[]){


  fx_init(argc,argv);
  Matrix q_matrix;
  Matrix r_matrix;
  Vector rset_weights;

  //Reading parameters and loading data

  struct datanode* regression_module =
    fx_submodule(NULL, "regression", "regression_module");
  
  // The reference data file is a required parameter.
  const char* reference_file_name = fx_param_str_req(regression_module, "data");
  
  // The query data file defaults to the references.
  const char* query_file_name =
    fx_param_str(regression_module, "query", reference_file_name);

  // flag for telling whether references are equal to queries
  bool queries_equal_references = 
    !strcmp(query_file_name, reference_file_name);
  
  // data::Load inits a matrix with the contents of a .csv or .arff.
  data::Load(reference_file_name, &r_matrix);
  
  if(queries_equal_references) {
    q_matrix.Alias(r_matrix);
  }

  else {
    data::Load(query_file_name, &q_matrix);
  }

  
  // confirm whether the user asked for scaling of the dataset
  if(!strcmp(fx_param_str(regression_module, "scaling", "none"), "range")) {
    DatasetScaler::ScaleDataByMinMax(q_matrix, r_matrix,
                                     queries_equal_references);
  }
 
  //Get the bandwidth for kernel calculations and the tolerance limit. 
  //Both default to 0.2

  double bandwidth=fx_param_double(regression_module,"bandwidth",0.15);
  double tau=fx_param_double(regression_module,"tau",0.15);
  printf("tau is %f\n",tau);

  //Get the weights for the reference set
  const char *rwfname=NULL;
  if (fx_param_exists (NULL, "dwgts")){
    
    //rwfname is the filename having the refernece weights
    rwfname =  fx_param_str (NULL, "dwgts", NULL);
  }
  
  if (rwfname != NULL){
    
    Dataset ref_weights;
    ref_weights.InitFromFile (rwfname);
    rset_weights.Copy (ref_weights.matrix ().GetColumnPtr (0), ref_weights.matrix ().n_rows ());	//Note rset_weights_ is a vector of weights
    
  }
  
  else{
    
    rset_weights.Init (r_matrix.n_cols ());
    rset_weights.SetAll (1);
    
  }

  //Get the length of the leaf. Defaulted to 2

  index_t leaf_length=fx_param_int(regression_module,"leaf_length",3);

  if(!strcmp(fx_param_str(regression_module,"kernel","gaussian"),"gaussian")){

    FastRegression<GaussianKernel> fast_regression;
   
    char *criteria;
    criteria=(char*)malloc(40*sizeof(char));
    strcpy(criteria,fx_param_str(regression_module,"criteria","fnorm"));
    
    fast_regression.Init(q_matrix, r_matrix, bandwidth, tau, leaf_length, rset_weights, criteria);
    fast_regression.Compute();



    //Lets do naive calculations too............
  
    //Lets first declare an object of the naive type
    printf("FAST CALCULATIONS ALL DONE");
    ArrayList<index_t> old_from_new_r;
    old_from_new_r.Copy(fast_regression.get_old_from_new_r());

    //Get the fast regression estimates.We shall use it to compare the
    //accuracy

    Vector fast_regression_estimate;
    fast_regression_estimate.Copy(fast_regression.get_regression_estimate());

    ArrayList<Matrix> fast_b_twy_estimates;
    ArrayList<Matrix> fast_b_twb_estimates;

    fast_b_twy_estimates.Init(q_matrix.n_cols());
    fast_b_twb_estimates.Init(q_matrix.n_cols());

    for(index_t q=0;q<q_matrix.n_cols();q++){

      fast_b_twy_estimates[q].Alias(fast_regression.get_b_twy_estimates(q));
      fast_b_twb_estimates[q].Alias(fast_regression.get_b_twb_estimates(q));
    }
    
    NaiveCalculation<GaussianKernel> naive;
    naive.Init(q_matrix,r_matrix,old_from_new_r,bandwidth,rset_weights);
    naive.Compute();
    naive.ComputeMaximumRelativeError
      (fast_b_twy_estimates,fast_b_twb_estimates,criteria);
    naive.CompareFastWithNaive(fast_regression_estimate);
   

    //Will need to verify if this is fine to do...

 

    fx_done();
  }
}





/**
 * @file cart_driver.c
 *
 * Main method for implementing classification and regression 
 * tree class.
 * @see cartree.h
 */

#include "fastlib/fastlib.h"
#include "cartree.h"

const fx_entry_doc root_entries[] = {
  {"target", FX_PARAM, FX_INT, NULL,
  "Target Variable, if in data file \n"},
  {"data", FX_REQUIRED, FX_STR,  NULL,
   "Test data file \n"},
  {"labels", FX_PARAM, FX_STR, NULL,
   "Labels for classification \n"},
  {"alpha", FX_PARAM, FX_DOUBLE, NULL,
   "Pruning criterion \n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc cart_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc root_doc = {
  root_entries, cart_submodules,
  "CART Parameters \n"
};

int main(int argc, char *argv[]){
  fx_module *root = fx_init(argc, argv, &root_doc);

  const char* fp;
  fp = fx_param_str_req(NULL, "data");
  const char* fl;
  fl = fx_param_str(NULL, "labels", " ");
 
  FILE *classifications;
  classifications = fopen("results.csv", "w+");
  int target_variable, points;
  target_variable = fx_param_int(NULL, "target", 0);
  double alpha;
  alpha = fx_param_double(0, "alpha", 1.0);
  int folds;
  folds = fx_param_int(NULL, "folds", 10);
  
  Vector alpha_error;
  alpha_error.Init(3600);
  alpha_error.SetZero();

  for (int i = 0; i < folds; i++){
     Matrix data_mat;
     TrainingSet data;
     Vector firsts;  
    // Read in data
    if (!strcmp(" ", fl)){   
      data_mat.Init(1,1);
      data.Init(fp, firsts);
      if (target_variable >= data.GetFeatures()){
	target_variable = data.GetFeatures() - 1;
      }
      if (target_variable < 0){
	target_variable = 0;
      }
    } else {      
      data.InitLabels(fp);  
      data_mat.Init(data.GetFeatures()+1, data.GetPointSize());
      data.InitLabels2(fl, firsts, &data_mat);
      printf("Target Variable: %d \n", target_variable);
      target_variable = data.GetFeatures()-1;    
    }
  
    points = data.GetPointSize();

    // Rearrange Data to facilitate building / validation
    int start, stop;
    start = (int)(i*points / folds);
    stop =  (int)((i+1)*points / folds);
    Vector split;
    split.Init(points);
    split.SetZero();
    for (int j = start; j < stop; j++){
      split[j] = 1.0;
    }
    Vector new_firsts, foo;
    data.MatrixPartition(0, points, split, firsts, &foo, &new_firsts);
    int CV_set = stop - start;     

    // Grow Tree
    CARTree tree;
    tree.Init(&data, new_firsts, CV_set, points, target_variable);
    tree.Grow();
    printf(" Tree has %d nodes. \n", 2*tree.GetNumNodes()-1);

  
    printf("Pruning with increasing alpha...\n");
    for (int k = 0; k < 3600; k++){
      // Prune tree with increasing alpha, measure success rate    
      tree.Prune(0.5*k);   
      int errors = 0;    
      for (int j = 0; j < CV_set; j++){
	double prediction = tree.Test(&data, j);
	errors = errors + (int)data.Verify(target_variable, prediction, j); 
      }
      alpha_error[k] = alpha_error[k] + errors;
    }
  }
  // Find optimal alpha
  double best_alpha = 0;
  int best_error = BIG_BAD_NUMBER;
  for (int k = 0; k < 3600; k++){
    if (alpha_error[k] < best_error){
      best_error = alpha_error[k];
      best_alpha = 0.5*k;
    }    
  }
  if (best_alpha > 0){
    alpha = best_alpha;
    printf("New Alpha: %f \n", alpha);
  }
  


  // Rebuild tree final time, using all data, prune using
  // C.V. determined alpha.
  Matrix data_mat;
  TrainingSet data;
  Vector firsts;  
  // Read in data
  if (!strcmp(" ", fl)){   
    data_mat.Init(1,1);
    data.Init(fp, firsts);
    if (target_variable >= data.GetFeatures()){
      target_variable = data.GetFeatures() - 1;
    }
    if (target_variable < 0){
      target_variable = 0;
    }
  } else {      
    data.InitLabels(fp);  
    data_mat.Init(data.GetFeatures()+1, data.GetPointSize());
    data.InitLabels2(fl, firsts, &data_mat);
    printf("Target Variable: %d \n", target_variable);
    target_variable = data.GetFeatures()-1;    
  }
  
  points = data.GetPointSize();  

  // Grow Tree
  CARTree tree;
  tree.Init(&data, firsts, 0, points, target_variable);
  tree.Grow();
  printf(" Tree has %d nodes. \n", 2*tree.GetNumNodes()-1);
  
  // Prune tree with increasing alpha, measure success rate
  printf("Pruning...\n");
  tree.Prune(alpha);
  printf("Tree has %d nodes. \n", 2*tree.GetNumNodes()-1);
  int errors = 0;    
  for (int j = 0; j < points; j++){
    double prediction = tree.Test(&data, j);
    errors = errors + (int)data.Verify(target_variable, prediction, j);      
  }
  printf("Error: %d \n", errors);   






  fclose(classifications);

  fx_done(root);

 

}

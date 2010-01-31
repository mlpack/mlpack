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
  "Dimension of target variable, if in data file \n"},
  {"data", FX_REQUIRED, FX_STR,  NULL,
   "Test data file \n"},
  {"labels", FX_PARAM, FX_STR, NULL,
   "Labels for classification, if in separate file \n"},
  {"alpha", FX_PARAM, FX_DOUBLE, NULL,
   "Pruning criterion. Input value ignored if using cross validation. \n"},
  {"folds", FX_PARAM, FX_DOUBLE, NULL,
   "Number of folds for cross validation. Use 0 to specify your own alpha \n"},
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
 
  FILE *result;
  result = fopen("tree.txt", "w+");
  int target_variable, points;
  target_variable = fx_param_int(NULL, "target", 0);
  double alpha;
  alpha = fx_param_double(0, "alpha", 1.0);
  int folds;
  folds = fx_param_int(0, "folds", 10);
  
  ArrayList<ArrayList<double> > alphas;
  alphas.Init(0);

  for (int i = 0; i < folds; i++){
    ArrayList<ArrayList<double> > fold_alpha;
    fold_alpha.Init(0);
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
    printf("\nFold %d \n---------------\n", i);
    CARTree tree;
    tree.Init(&data, new_firsts, CV_set, points, target_variable);
    tree.Grow();
    int max_nodes = tree.GetNumNodes();
   
    printf("Tree has %d nodes. \n", 2*max_nodes-1);

    
    printf("Pruning with increasing alpha...\n");
    double current_alpha = 0;
    for (int j = 0; j < CV_set; j++){
      tree.SetTestError(&data, j);
    }
    double errors = 0;
    
    while(tree.GetNumNodes() > 1 & current_alpha < BIG_BAD_NUMBER){      
      errors = tree.GetTestError();            
	fold_alpha.PushBack();
	int size_list = fold_alpha.size() - 1;
	fold_alpha[size_list].Init(2);
	fold_alpha[size_list][0] = current_alpha;
	fold_alpha[size_list][1] = errors;  
      current_alpha = tree.Prune(current_alpha) + 1.0e-8;      
    }
    
    // Merge list of alpha vs. error
    ArrayList<ArrayList<double> > new_alphas;
    new_alphas.Init(0);
    int master = 0, fold = 0, total = 0;     
    while(fold < fold_alpha.size() & master < alphas.size()){
      new_alphas.PushBack();
      new_alphas[total].Init(2);
      new_alphas[total][0] = min(alphas[master][0], fold_alpha[fold][0]);
      new_alphas[total][1] = alphas[master][1] + fold_alpha[fold][1];
      total++;
      if (alphas[master][0] == fold_alpha[fold][0]){
	fold++;
	master++;		  
      } else {
	if (alphas[master][0] < fold_alpha[fold][0]){
	  master++;	   	  
	} else {
	  fold++;	    
	}
      }     
    }             
    
    for (int k = master; k < alphas.size(); k++){	
      new_alphas.PushBack();
      new_alphas[total].Init(2);
      new_alphas[total][0] = alphas[k][0];
      new_alphas[total][1] = alphas[k][1] + fold_alpha[fold-1][1];
      total++;       
    }      
    
    for (int k = fold; k < fold_alpha.size(); k++){
      new_alphas.PushBack();      
      new_alphas[total].Init(2);      
      new_alphas[total][0] = fold_alpha[k][0];
      if (master == 0){
	new_alphas[total][1] = fold_alpha[k][1];
      } else {
	new_alphas[total][1] = alphas[master-1][1] + fold_alpha[k][1];
      }
      total++;
    }
    alphas = new_alphas;  
  }
  
  // Find optimal alpha
  double best_alpha = 0;
  int best_error = BIG_BAD_NUMBER;
  for (int k = 0; k < alphas.size(); k++){
    if (alphas[k][1] <= best_error){
      best_error = (int)alphas[k][1];
      best_alpha = alphas[k][0];
    }    
  }
  if (folds > 0){
    alpha = best_alpha;
    printf("\n CV-determined cost complexity: %5.2f \n", alpha);
  }
  
  
   printf("\nFinal Tree \n-------------- \n");
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
    target_variable = data.GetFeatures()-1;    
  }
  
  points = data.GetPointSize();  

  // Grow Tree
  CARTree tree;
  tree.Init(&data, firsts, 0, points, target_variable);
  tree.Grow();
 
  printf("Tree has %d nodes. \n", 2*tree.GetNumNodes()-1);
  
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

  // Write Tree to file
  tree.WriteTree(0, result); 



  fclose(result);

  fx_done(root);

 

}

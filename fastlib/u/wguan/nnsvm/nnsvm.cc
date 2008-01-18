#include "nnsvm.h"



int LoadData(Dataset* dataset, String datafilename){
    if (fx_param_exists(NULL, datafilename)) {
      // if a data file is specified, use it.
      if (!PASSED(dataset->InitFromFile(fx_param_str_req(NULL, datafilename)))) {
	fprintf(stderr, "Couldn't open the data file.\n");
	return 0;
      }
    }
    return 1;
}



int main(int argc, char *argv[]) {
  

  fx_init(argc, argv);

  String mode = fx_param_str_req(NULL, "mode");
  String kernel = fx_param_str_req(NULL, "kernel");  
  
  if (mode=="train"){
    fprintf(stderr, "Postivity Constrained SVM Training... \n");

    // Load training data
    Dataset trainset;
    if (LoadData(&trainset, "train_data") == 0) // TODO:param_req
      return 1;    
   
    // Begin Postivity Constrained SVM Training 
    datanode *svm_module = fx_submodule(fx_root, NULL, "svm");
    
    if (kernel == "linear") {
      SVM<SVMLinearKernel> svm;
      svm.InitTrain(trainset,2, svm_module);

  	
      timer *nnsvm_train_time = fx_timer(NULL, "nnsvm_train");

      fprintf(stderr, "nnsvm_train_time %g \n", nnsvm_train_time->total.micros / 1e6);
    }
    
    fprintf(stderr, "Postivity Constrained SVM Training Done! \n");
  }

  fx_done();

}


#include "fastlib/fastlib.h"
#include "test_utils.h"
#include "test_engine.h"


int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  //const char* mode = fx_param_str_req(NULL, "mode");
  /*
    if(strcmp(mode, "full") == 0) {
    HMM<Multinomial> hmm;
    ArrayList<GenMatrix<int> > sequences;
    GenVector<int> labels;
    LoadOneSynthHMMAndSequences(&hmm, &sequences, &labels);
    TestHMMFisherKernelClassification(hmm, sequences, labels);
    }
    else if(strcmp(mode, "kfold") == 0) {
  */
  const char* model_classes = fx_param_str_req(NULL, "model_classes");
  if((strcmp(model_classes, "pos") == 0)
     || (strcmp(model_classes, "neg") == 0)) {
    ArrayList<HMM<DiagGaussian> > kfold_hmms;
    ArrayList<GenMatrix<double> > sequences;
    GenVector<int> labels;
    int n_folds = fx_param_int(NULL, "n_folds", 10);
    LoadKFoldHMMAndSequences(n_folds, &kfold_hmms, &sequences, &labels);
    TestHMMFisherKernelClassificationKFold(n_folds,
					   kfold_hmms,
					   sequences, labels);
  }
  else if(strcmp(model_classes, "both") == 0) {
    ArrayList<HMM<DiagGaussian> > kfold_class1_hmms;
    ArrayList<HMM<DiagGaussian> > kfold_class0_hmms;
    ArrayList<GenMatrix<double> > sequences;
    GenVector<int> labels;
    int n_folds = fx_param_int(NULL, "n_folds", 10);
    LoadKFoldHMMPairAndSequences(n_folds,
				      &kfold_class1_hmms, &kfold_class0_hmms,
				      &sequences, &labels);
    TestHMMFisherKernelClassificationKFold(n_folds,
					   kfold_class1_hmms, kfold_class0_hmms,
					   sequences, labels);
  }
  else {
    FATAL("Error: For k-fold cross-validation, parameter 'model_classes' must be set to \"pos\", \"neg\", or \"both\". Exiting...");
  }
  /*
    }
    else {
    FATAL("Error: Parameter 'mode' must be set to \"kfold\". Exiting...");
    }
  */    
  
  fx_done(fx_root);
}

#include "fastlib/fastlib.h"
#include "test_dna_utils.h"
#include "test_engine.h"


int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  const char* mode = fx_param_str_req(NULL, "mode");
  if(strcmp(mode, "full") == 0) {
    HMM<Multinomial> hmm;
    ArrayList<GenMatrix<int> > sequences;
    GenVector<int> labels;
    LoadOneDNAHMMAndSequences(&hmm, &sequences, &labels);
    //TestHMMLatMMKClassification(hmm, sequences, labels);
  }
  else if(strcmp(mode, "kfold") == 0) {
    const char* model_classes = fx_param_str_req(NULL, "model_classes");
    if((strcmp(model_classes, "exons") == 0)
       || (strcmp(model_classes, "introns") == 0)) {
      ArrayList<HMM<Multinomial> > kfold_hmms;
      ArrayList<GenMatrix<int> > sequences;
      GenVector<int> labels;
      int n_folds = fx_param_int(NULL, "n_folds", 10);
      LoadKFoldDNAHMMAndSequences(n_folds, &kfold_hmms, &sequences, &labels);
      TestHMMLatMMK2ClassificationKFold(n_folds, kfold_hmms, sequences, labels);
    }
    else if(strcmp(model_classes, "both") == 0) {
      ArrayList<HMM<Multinomial> > kfold_exon_hmms;
      ArrayList<HMM<Multinomial> > kfold_intron_hmms;
      ArrayList<GenMatrix<int> > sequences;
      GenVector<int> labels;
      int n_folds = fx_param_int(NULL, "n_folds", 10);
      LoadKFoldDNAHMMPairAndSequences(n_folds,
				      &kfold_exon_hmms, &kfold_intron_hmms,
				      &sequences, &labels);
      TestHMMLatMMK2ClassificationKFold(n_folds,
					kfold_exon_hmms, kfold_intron_hmms,
					sequences, labels);
    }
    else {
      FATAL("Error: For k-fold cross-validation, parameter 'model_classes' must be set to \"exons\", \"introns\", or \"both\". Exiting...");
    }
  }
  else {
    FATAL("Error: Parameter 'mode' must be set to \"full\" or \"kfold\". Exiting...");
  }    
  
  fx_done(fx_root);
}

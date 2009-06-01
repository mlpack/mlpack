#include "fastlib/fastlib.h"
#include "test_dna_utils.h"
#include "test_engine.h"
//#include "utils.h"


int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  const char* mode = fx_param_str_req(NULL, "mode");
  if(strcmp(mode, "full") == 0) {
    HMM<Multinomial> hmm;
    ArrayList<GenMatrix<int> > sequences;
    GenVector<int> labels;
    LoadOneDNAHMMAndSequences(&hmm, &sequences, &labels);
    /*
      ReadInOTObject("frozen_dna_one_hmm_topo4_model_exons", &hmm);
      ReadInOTObject("frozen_dna_labels", &labels);

      const char* exons_filename = "exons_small.dat";
      const char* introns_filename = "introns_small.dat";

      LoadVaryingLengthData(exons_filename, &sequences);

      ArrayList<GenMatrix<int> > intron_sequences;
      LoadVaryingLengthData(introns_filename, &intron_sequences);
      sequences.AppendSteal(&intron_sequences);

      double val = FisherKernel(hmm, sequences[0], sequences[1]);
      printf("fisher kernel = %f\n", val);
    */
  
    TestHMMFisherKernelClassification(hmm, sequences, labels);
  }
  else if(strcmp(mode, "kfold") == 0) {
    ArrayList<HMM<Multinomial> > kfold_hmms;
    ArrayList<GenMatrix<int> > sequences;
    GenVector<int> labels;
    int n_folds = fx_param_int(NULL, "n_folds", 10);
    LoadKFoldDNAHMMAndSequences(n_folds, &kfold_hmms, &sequences, &labels);
    TestHMMFisherKernelClassificationKFold(n_folds, kfold_hmms, sequences, labels);
  }
  else {
    FATAL("Error: Parameter 'mode' must be set to \"full\" or \"kfold\". Exiting...");
  }    
  
  fx_done(fx_root);
}

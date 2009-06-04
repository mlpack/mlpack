#include "fastlib/fastlib.h"
#include "test_synth_utils.h"
#include "test_engine.h"
#include "utils.h"


int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  srand48(time(0));
  srand(time(0));


  ArrayList<HMM<Multinomial> > kfold_class1_hmms;
  ArrayList<HMM<Multinomial> > kfold_class0_hmms;
  ArrayList<GenMatrix<int> > sequences;
  GenVector<int> labels;
  int n_folds = fx_param_int(NULL, "n_folds", 10);
  LoadKFoldSynthHMMPairAndSequences(n_folds,
				  &kfold_class1_hmms, &kfold_class0_hmms,
				  &sequences, &labels);
  TestHMMBayesClassificationKFold(n_folds, kfold_class1_hmms, kfold_class0_hmms,
				  sequences, labels);

  fx_done(fx_root);
}

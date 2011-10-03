/* USAGE

--exp=(name of experiment) (this determines the filename of the loaded HMMs/labels)

--n_states_pos=j
--n_states_neg=k

--data_pos= (positive class data filename)
--data_neg= (negative class data filename)

*/


#include "fastlib/fastlib.h"
#include "test_utils.h"
#include "test_engine.h"
#include "utils.h"


int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  srand48(time(0));
  srand(time(0));


  ArrayList<HMM<DiagGaussian> > kfold_class1_hmms;
  ArrayList<HMM<DiagGaussian> > kfold_class0_hmms;
  ArrayList<GenMatrix<double> > sequences;
  GenVector<int> labels;
  int n_folds = fx_param_int(NULL, "n_folds", 10);
  LoadKFoldHMMPairAndSequences(n_folds,
			       &kfold_class1_hmms, &kfold_class0_hmms,
			       &sequences, &labels);
  TestHMMBayesClassificationKFold(n_folds, kfold_class1_hmms, kfold_class0_hmms,
				  sequences, labels);
  
  fx_done(fx_root);
}

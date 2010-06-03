#ifndef KFOLD_SPLITTER_H
#define KFOLD_SPLITTER_H

#include "fastlib/fastlib_int.h"

class KFoldSplitter {

  private:

    /** pointer to the entire dataset */
    Dataset data_;

    /** number of k-folds */
    int n_folds_;

    /** whether to do randomized selection */
    bool randomized_;

    void SaveTrainTest_(int i_folds, const Dataset& train,
                        const Dataset& test) const {
      String train_name;
      String test_name;
      const char *file_name = fx_param_str_req(NULL, "data");

      train_name.InitSprintf("%s_train_%d.csv", file_name, i_folds);
      test_name.InitSprintf("%s_test_%d.csv", file_name, i_folds);

      train.WriteCsv(train_name);
      test.WriteCsv(test_name);
    }

  public:

    /** initialize */
    void Init() {
      const char *file_name = fx_param_str_req(NULL, "data");
      data_.InitFromFile(file_name);
      n_folds_ = fx_param_int_req(NULL, "num_fold");
      randomized_ = fx_param_exists(NULL, "random");
    }

    void Split() {
      ArrayList<index_t> permutation;

      if (randomized_) {
        math::MakeRandomPermutation(data_.n_points(), &permutation);
      }
      else {
        math::MakeIdentityPermutation(data_.n_points(), &permutation);
      }

      for (int i_folds = 0; i_folds < n_folds_; i_folds++) {
        Dataset test;
        Dataset train;
        data_.SplitTrainTest(n_folds_, i_folds, permutation, &train, &test);

        SaveTrainTest_(i_folds, train, test);
      }

    }

};

#endif

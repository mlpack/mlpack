#ifndef MULTITREE_UTILITY_H
#define MULTITREE_UTILITY_H

class MultiTreeUtility {

 public:

  static void ShuffleAccordingToQueryPermutation
  (Matrix &v, const ArrayList<index_t> &permutation) {
    
    Matrix v_tmp;
    la::TransposeInit(v, &v_tmp);
    for(index_t i = 0; i < v.n_rows(); i++) {
      Vector column_vector;
      v_tmp.MakeColumnVector(i, &column_vector);
      ShuffleAccordingToQueryPermutation(column_vector, permutation);
    }
    la::TransposeOverwrite(v_tmp, &v);
  }

  /** @brief Shuffles a vector according to a given permutation.
   *
   *  @param v The vector to be shuffled.
   *  @param permutation The permutation.
   */
  static void ShuffleAccordingToPermutation
  (Vector &v, const ArrayList<index_t> &permutation) {
    
    Vector v_tmp;
    v_tmp.Init(v.length());
    for(index_t i = 0; i < v_tmp.length(); i++) {
      v_tmp[i] = v[permutation[i]];
    }
    v.CopyValues(v_tmp);
  }

  static void ShuffleAccordingToQueryPermutation
  (Vector &v, const ArrayList<index_t> &permutation) {
    
    Vector v_tmp;
    v_tmp.Init(v.length());
    for(index_t i = 0; i < v_tmp.length(); i++) {
      v_tmp[permutation[i]] = v[i];
    }
    v.CopyValues(v_tmp);
  }

  /** @brief Shuffles a vector according to a given permutation.
   *
   *  @param v The vector to be shuffled.
   *  @param permutation The permutation.
   */
  static void ShuffleAccordingToPermutation
  (Matrix &v, const ArrayList<index_t> &permutation) {
    
    for(index_t c = 0; c < v.n_cols(); c++) {
      Vector column_vector;
      v.MakeColumnVector(c, &column_vector);
      ShuffleAccordingToPermutation(column_vector, permutation);
    }
  }

};

#endif

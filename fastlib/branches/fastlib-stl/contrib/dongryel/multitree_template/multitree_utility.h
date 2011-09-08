/** @file multitree_utility.h
 *
 *  The common utility for shuffling and reshuffling results.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MULTITREE_UTILITY_H
#define MULTITREE_UTILITY_H

class MultiTreeUtility {

 public:

  /** @brief Implements Knuth's random combination selector algorithm.
   */
  static void RandomCombination(size_t begin_inclusive, 
				size_t end_exclusive, size_t num_times, 
				size_t *output) {
    
    size_t t = 0, m = 0;
    size_t range = end_exclusive - begin_inclusive;

    do {
      double u = math::Random();
      if((range - t) * u >= num_times - m) {
	t++;
	continue;
      }
      
      output[m] = begin_inclusive + t;

      m++;
      t++;
      if(m >= num_times) {
	break;
      }
    } while(1);
  }

  template<typename Tree>
  static void RandomTuple(const ArrayList<Tree *> &nodes,
			  ArrayList<size_t> &random_permutation) {

    for(size_t block_pointer = 0; block_pointer < nodes.size(); ) {
      
      // Figure out how many blocks of nodes are equal.
      size_t saved_block_pointer = block_pointer;
      do {
	block_pointer++;
      } while(nodes[block_pointer] == nodes[saved_block_pointer]);

      RandomCombination(nodes[saved_block_pointer]->begin(),
			nodes[saved_block_pointer]->end(),
			block_pointer - saved_block_pointer,
			random_permutation.begin() + saved_block_pointer);
      
    }
  }

  static void ShuffleAccordingToQueryPermutation
  (Matrix &v, const ArrayList<size_t> &permutation) {
    
    Matrix v_tmp;
    la::TransposeInit(v, &v_tmp);
    for(size_t i = 0; i < v.n_rows(); i++) {
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
  (Vector &v, const ArrayList<size_t> &permutation) {
    
    Vector v_tmp;
    v_tmp.Init(v.length());
    for(size_t i = 0; i < v_tmp.length(); i++) {
      v_tmp[i] = v[permutation[i]];
    }
    v.CopyValues(v_tmp);
  }

  static void ShuffleAccordingToQueryPermutation
  (Vector &v, const ArrayList<size_t> &permutation) {
    
    Vector v_tmp;
    v_tmp.Init(v.length());
    for(size_t i = 0; i < v_tmp.length(); i++) {
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
  (Matrix &v, const ArrayList<size_t> &permutation) {
    
    for(size_t c = 0; c < v.n_cols(); c++) {
      Vector column_vector;
      v.MakeColumnVector(c, &column_vector);
      ShuffleAccordingToPermutation(column_vector, permutation);
    }
  }

};

#endif

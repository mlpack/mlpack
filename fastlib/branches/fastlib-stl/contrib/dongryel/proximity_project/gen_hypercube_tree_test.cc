#include "fastlib/fastlib.h"
#include "fastlib/tree/statistic.h"
#include "general_spacetree.h"
#include "contrib/dongryel/fast_multipole_method/fmm_stat.h"
#include "contrib/dongryel/pca/pca.h"
#include "subspace_stat.h"
#include "gen_hypercube_tree.h"
#include "mlpack/kde/dataset_scaler.h"

int BitInterleaving(const GenVector<unsigned int> &indices) {
  
  int result = 0;
  unsigned int offset = 0;
  GenVector<unsigned int> indices_copy;
  indices_copy.Copy(indices);
  
  do {
    unsigned int sum = 0;
    for(index_t d = 0; d < indices_copy.length(); d++) {
      sum += indices_copy[d];
    }
    if(sum == 0) {
      break;
    }
    
    for(index_t d = 0; d < indices_copy.length(); d++) {
      result += (indices_copy[d] % 2) << 
	(indices_copy.length() - d - 1 + offset);
      indices_copy[d] = indices_copy[d] >> 1;
    }
    offset += indices_copy.length();
    
  } while(true);
  
  return result;
}

void BitDeinterleaving(unsigned int index, unsigned int level,
		       GenVector<unsigned int> &indices) {
  
  for(index_t d = 0; d < indices.length(); d++) {
    indices[d] = 0;
  }
  unsigned int loop = 0;
  while(index > 0 || level > 0) {
    for(index_t d = indices.length() - 1; d >= 0; d--) {
      indices[d] = (1 << loop) * (index % 2) + indices[d];
      index = index >> 1;
    }      
    level--;
    loop++;
  }
}

void RecursivelyChooseIndex(const GenVector<unsigned int> &lower_limit,
			    const GenVector<unsigned int> &exclusion_index,
			    const GenVector<unsigned int> &upper_limit,
			    GenVector<unsigned int> &chosen_index, int level,
			    bool valid_combination,
			    ArrayList<unsigned int> &neighbor_indices) {
  
  if(level < lower_limit.length()) {
    
    // Choose the lower index.
    chosen_index[level] = lower_limit[level];
    RecursivelyChooseIndex(lower_limit, exclusion_index, upper_limit,
			   chosen_index, level + 1, valid_combination ||
			   (chosen_index[level] != exclusion_index[level]),
			   neighbor_indices);
    
    // Choose the exclusion index.
    chosen_index[level] = exclusion_index[level];
    RecursivelyChooseIndex(lower_limit, exclusion_index, upper_limit,
			   chosen_index, level + 1, valid_combination ||
			   (chosen_index[level] != exclusion_index[level]),
			   neighbor_indices);
    
    // Choose the upper index.
    chosen_index[level] = upper_limit[level];
    RecursivelyChooseIndex(lower_limit, exclusion_index, upper_limit,
			   chosen_index, level + 1, valid_combination ||
			   (chosen_index[level] != exclusion_index[level]),
			   neighbor_indices);
  }
  else {
    
    // If the chosen index is not equal to the exclusion index, then
    // add the node number to the list.
    if(valid_combination) {
      neighbor_indices.PushBackCopy(BitInterleaving(chosen_index));
    }
  }
}

void FindNeighborsInNonAdaptiveGenHypercubeTree
(unsigned int index, index_t level, index_t dimension, 
 ArrayList<unsigned int> &neighbor_indices) {
  
  // First, de-interleave the box index.
  GenVector<unsigned int> tmp_vector, lower_limit, upper_limit;
  tmp_vector.Init(dimension);
  lower_limit.Init(dimension);
  upper_limit.Init(dimension);
  BitDeinterleaving(index, level, tmp_vector);
  
  for(index_t d = 0; d < dimension; d++) {
    lower_limit[d] = std::max(tmp_vector[d] - 1, (unsigned int) 0);
    upper_limit[d] = std::min(tmp_vector[d] + 1, 
			      (unsigned int) ((1 << level) - 1));
  }
  
  GenVector<unsigned int> chosen_index;
  chosen_index.Init(dimension);
  RecursivelyChooseIndex(lower_limit, tmp_vector, upper_limit, chosen_index,
			 0, false, neighbor_indices);
}

int main(int argc, char *argv[]) {
 
  fx_init(argc, argv, NULL);
  const char *fname = fx_param_str(NULL, "data", NULL);
  Dataset dataset_incoming;
  dataset_incoming.InitFromFile(fname);
  Matrix dataset;
  dataset.Own(&(dataset_incoming.matrix()));

  int leaflen = fx_param_int(NULL, "leaflen", 30);

  printf("Constructing the tree...\n");
  fx_timer_start(NULL, "generalized_hypercube_tree_build");

  ArrayList<Matrix *> matrices;
  matrices.Init();
  matrices.PushBackCopy(&dataset);

  ArrayList< ArrayList<index_t> > old_from_new;
  ArrayList< ArrayList<proximity::GenHypercubeTree< EmptyStatistic<Matrix> > *> > nodes_in_each_level;
  proximity::GenHypercubeTree<EmptyStatistic<Matrix> > *root;
  root = proximity::MakeGenHypercubeTree
    (matrices, leaflen, 4, &nodes_in_each_level, &old_from_new);
  
  fx_timer_stop(NULL, "generalized_hypercube_tree_build");

  for(index_t i = 0; i < nodes_in_each_level.size(); i++) {
    for(index_t j = 0; j < nodes_in_each_level[i].size(); j++) {
      printf("%u ", (nodes_in_each_level[i][j])->node_index());
    }
    printf("\n");
  }

  printf("Finished constructing the tree...\n");

  // Print the tree.
  root->Print();

  // Clean up the memory used by the tree...
  delete root;

  fx_done(fx_root);
  return 0;
}

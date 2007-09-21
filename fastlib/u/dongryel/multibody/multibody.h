#ifndef MULTIBODY_H
#define MULTIBODY_H

#include "u/dongryel/series_expansion/farfield_expansion.h"
#include "u/dongryel/series_expansion/local_expansion.h"
#include "u/dongryel/series_expansion/series_expansion_aux.h"


template<typename TKernel, typename TKernelDerivative>
class MultitreeMultibody {
  
public:
  
  class MultibodyStat {
    
    /** Summed up potential for query points in this node */
    double potential_;

    /**
     * Extra amount of error that can be spent for the query points in
     * this node.
     */
    double extra_token_;

    /**
     * Far field expansion created by the reference points in this node.
     */
    FarFieldExpansion<TKernel, TKernelDerivative> farfield_expansion_;

    /**
     * Local expansion stored in this node.
     */
    LocalExpansion<TKernel, TKernelDerivative> local_expansion_;

    /** Initialize the statistics */
    void Init(double bandwidth, const Vector& center, 
	      SeriesExpansionAux *sea) {
      
      potential_ = 0;
      extra_token_ = 0;
      farfield_expansion_.Init(bandwidth, center, sea);
      local_expansion_.Init(bandwidth, center, sea);
    }
  };

  typedef BinarySpaceTree<DHrectBound<2>, Matrix, MultibodyStat> 
    MultibodyTree;
  
  MultitreeMultibody() {}
  
  ~MultitreeMultibody() { delete root; }

private:

  /** pointer to the root of the tree */
  MultibodyTree *root;

  /** Initialize the tree */
  void Init(const Matrix& data) {
    fx_timer_start(NULL, "tree_d");
    tree::LoadKdTree(fx_submodule(NULL, "r", "read_r"), &data, &root, NULL);
    fx_timer_stop(NULL, "tree_d");
  }
};

#endif

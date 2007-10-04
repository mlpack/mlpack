#ifndef KDE_H
#define KDE_H

#include "fastlib/fastlib_int.h"
#include "u/dongryel/series_expansion/farfield_expansion.h"
#include "u/dongryel/series_expansion/local_expansion.h"
#include "u/dongryel/series_expansion/kernel_derivative.h"

template<typename TKernel, typename TKernelDerivative>
class KdeStat {
 public:

  /**
   * Far field expansion created by the reference points in this node.
   */
  FarFieldExpansion<TKernel, TKernelDerivative> farfield_expansion_;
  
  /**
   * Local expansion stored in this node.
   */
  LocalExpansion<TKernel, TKernelDerivative> local_expansion_;

  // getters and setters
  FarFieldExpansion<TKernel, TKernelDerivative> &get_farfield_coeffs() {
    return farfield_expansion_;
  }

  /** Initialize the statistics */
  void Init() {
  }

  void Init(double bandwidth, SeriesExpansionAux *sea) {
    farfield_expansion_.Init(bandwidth, sea);
    local_expansion_.Init(bandwidth, sea);
  }

  void Init(const Matrix& dataset, index_t &start, index_t &count) {
    Init();
  }

  void Init(const Matrix& dataset, index_t &start, index_t &count,
            const KdeStat& left_stat,
            const KdeStat& right_stat) {
    Init();
  }

  void Init(double bandwidth, const Vector& center,
            SeriesExpansionAux *sea) {

    farfield_expansion_.Init(bandwidth, center, sea);
    local_expansion_.Init(bandwidth, center, sea);
  }

  KdeStat() { }

  ~KdeStat() {}

};

template<typename TKernel, typename TKernelDerivative>
class FastKde {
  
 private:
  
  // member variables
  typedef BinarySpaceTree<DHrectBound<2>, Matrix,
    KdeStat<TKernel, TKernelDerivative> > Tree;
  
  /** query dataset */
  Matrix qset_;

  /** query tree */
  Tree *qroot_;

  /** reference dataset */
  Matrix rset_;

  /** reference tree */
  Tree *rroot_;

  /** list of kernels to evaluate */
  ArrayList<TKernel> kernels_;

  /** accuracy parameter */
  double tau_;
  
  // member functions
  void FKde() {
    
  }
  
 public:
  
  FastKde() {}

  ~FastKde() {}

  void Compute(double tau) {

    tau_ = tau;

    
  }

  void Init() {
    
    Dataset ref_dataset;

    // read in the number of points owned by a leaf
    int leaflen = fx_param_int(NULL, "leaflen", 20);

    // read the datasets
    const char *rfname = fx_param_str(NULL, "data", NULL);
    const char *qfname = fx_param_str(NULL, "query", NULL);
    
    // construct query and reference trees
    ref_dataset.InitFromFile(rfname);
    rset_.Own(&(ref_dataset.matrix()));
    
    fx_timer_start(NULL, "tree_d");
    rroot_ = tree::MakeKdTreeMidpoint<Tree>(rset_, leaflen, NULL);

    if(qfname == NULL) {
      qset_.Alias(rset_);
      qroot_ = rroot_;
    }
    else {
      Dataset query_dataset;
      query_dataset.InitFromFile(qfname);
      qset_.Own(&(query_dataset.matrix()));
      qroot_ = tree::MakeKdTreeMidpoint<Tree>(qset_, leaflen, NULL);
    }
    fx_timer_stop(NULL, "tree_d");

    
  }

};

#endif

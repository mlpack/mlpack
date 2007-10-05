#ifndef KDE_H
#define KDE_H

#include "fastlib/fastlib_int.h"
#include "u/dongryel/series_expansion/farfield_expansion.h"
#include "u/dongryel/series_expansion/local_expansion.h"
#include "u/dongryel/series_expansion/kernel_derivative.h"

template<typename TKernel>
class NaiveKde {

 private:
  
  /** query dataset */
  Matrix qset_;
  
  /** reference dataset */
  Matrix rset_;

  /** kernel */
  TKernel kernel_;

  /** computed densities */
  Vector densities_;
  
 public:
  
  void Compute() {
    
    // compute unnormalized sum
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      
      const double *q_col = qset_.GetColumnPtr(q);
      for(index_t r = 0; r < rset_.n_cols(); r++) {
	const double *r_col = rset_.GetColumnPtr(r);
	double dsqd = la::DistanceSqEuclidean(qset_.n_rows(), q_col, r_col);
	
	densities_[q] += kernel_.EvalUnnormOnSq(dsqd);
      }
    }
    
    // then normalize it
    double norm_const = kernel_.CalcNormConstant(qset_.n_rows());
    for(index_t q = 0; q < qset_.n_cols(); q++) {
      densities_[q] /= norm_const;
    }
  }

  void Init() {
    densities_.SetZero();
  }

  void Init(Matrix &qset, Matrix &rset, double bandwidth) {

    // get datasets
    qset_.Own(qset);
    rset_.Own(rset);

    // get bandwidth
    kernel_.Init(bandwidth);
    
    // allocate density storage
    densities_.Init();
    densities_.SetZero();
  }

};

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

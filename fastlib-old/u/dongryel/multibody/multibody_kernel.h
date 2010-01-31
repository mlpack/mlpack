#ifndef MULTIBODY_KERNEL_H
#define MULTIBODY_KERNEL_H

#include "fastlib/fastlib_int.h"
#include "u/dongryel/series_expansion/kernel_aux.h"

class GaussianThreeBodyKernel {
  
 private:
  GaussianKernel kernel_;
  
  Matrix distmat_;

 public:

  GaussianThreeBodyKernel() {}
  
  ~GaussianThreeBodyKernel() {}

  // getters and setters
  double bandwidth_sq() const { return kernel_.bandwidth_sq(); }

  const Matrix &pairwise_dsqd() const { return distmat_; }

  void Init(double bandwidth_in) {
    kernel_.Init(bandwidth_in);
    distmat_.Init(3, 3);
  }
  
  int order() {
    return 3;
  }

  double EvalUnnormOnSqOnePair(double sqdist) const {
    return kernel_.EvalUnnormOnSq(sqdist);
  }

  void EvalUnnormOnSq(const Matrix &sqdists, double *neg, double *pos) const {
    
    *pos = 1;
    *neg = 0;

    for(index_t i = 0; i < sqdists.n_cols(); i++) {
      for(index_t j = i + 1; j < sqdists.n_cols(); j++) {

	(*pos) *= kernel_.EvalUnnormOnSq(sqdists.get(i, j));
      }
    }
  }

  void EvalMinMax(double *negmin, double *negmax,
		  double *posmin, double *posmax) const {

    *negmin = *negmax = 0;
    *posmin = 1.0;
    *posmax = 1.0;

    for(index_t i = 0; i < 3; i++) {
      for(index_t j = i + 1; j < 3; j++) {
	*posmin = (*posmin) * kernel_.EvalUnnormOnSq(distmat_.get(j, i));
	*posmax = (*posmax) * kernel_.EvalUnnormOnSq(distmat_.get(i, j));
      }
    }
  }

  const Matrix &EvalMinMaxDsqds
    (const ArrayList<DHrectBound<2> *> &node_bounds) {

    int num_nodes = node_bounds.size();

    for(index_t i = 0; i < num_nodes - 1; i++) {
      DHrectBound<2> *node_i_bound = node_bounds[i];

      for(index_t j = i + 1; j < num_nodes; j++) {
        DHrectBound<2> *node_j_bound = node_bounds[j];
        double dmin = node_i_bound->MinDistanceSq(*node_j_bound);
        double dmax = node_i_bound->MaxDistanceSq(*node_j_bound);

        distmat_.set(i, j, dmin);
        distmat_.set(j, i, dmax);
      }
    }
    
    return distmat_;
  }

  void Eval(const Matrix &data, const ArrayList<int> &indices, double *neg,
	    double *pos) {
    
    for(index_t i = 0; i < indices.size(); i++) {
      const double *i_col = data.GetColumnPtr(indices[i]);
      for(index_t j = i + 1; j < indices.size(); j++) {
	const double *j_col = data.GetColumnPtr(indices[j]);
	distmat_.set(i, j, la::DistanceSqEuclidean(data.n_rows(), i_col,
						   j_col));
      }
    }
    
    EvalUnnormOnSq(distmat_, neg, pos);
  }

  void EvalNodes(const ArrayList<DHrectBound<2> *> &node_bounds, 
		 double *negmin, double *negmax, double *posmin, 
		 double *posmax) {
    
    int num_nodes = node_bounds.size();

    for(index_t i = 0; i < num_nodes - 1; i++) {
      DHrectBound<2> *node_i_bound = node_bounds[i];

      for(index_t j = i + 1; j < num_nodes; j++) {
        DHrectBound<2> *node_j_bound = node_bounds[j];
        double dmin = node_i_bound->MinDistanceSq(*node_j_bound);
        double dmax = node_i_bound->MaxDistanceSq(*node_j_bound);

        distmat_.set(i, j, dmin);
        distmat_.set(j, i, dmax);
      }
    }
    
    EvalMinMax(negmin, negmax, posmin, posmax);
  }

};

class AxilrodTellerKernel {
    
 private:
  
  Matrix distmat_;

  static const double AXILROD_TELLER_COEFF = 1e-18;

 public:

  AxilrodTellerKernel() {}
  
  ~AxilrodTellerKernel() {}

  // getters and setters
  double bandwidth_sq() const { return 1; }

  const Matrix &pairwise_dsqd() const { return distmat_; }
  
  void Init(double bandwidth_in) {
    distmat_.Init(3, 3);
  }
  
  int order() {
    return 3;
  }

  double EvalUnnormOnSqOnePair(double sqdist) const {

    // this is a place holder, needs to be corrected...
    return 0;
  }

  void EvalUnnormOnSq(const Matrix &sqdists, double *neg, double *pos) const {

    *neg = -0.375 * 
      (sqdists.get(0, 1) * sqdists.get(0, 1) * sqdists.get(0, 1) + 
       sqdists.get(0, 2) * sqdists.get(0, 2) * sqdists.get(0, 2) +
       sqdists.get(1, 2) * sqdists.get(1, 2) * sqdists.get(1, 2)) /
      pow(sqdists.get(0, 1) * sqdists.get(0, 2) * distmat_.get(1, 2), 2.5);
    *pos = (3 * sqdists.get(0, 1) * sqdists.get(0, 1) *
	    (sqdists.get(0, 2) + sqdists.get(1, 2)) +
	    3 * sqdists.get(0, 2) * sqdists.get(1, 2) *
	    (sqdists.get(0, 2) + sqdists.get(1, 2)) +
	    sqdists.get(0, 1) *
	    (3 * sqdists.get(0, 2) * sqdists.get(0, 2) +
	     2 * sqdists.get(0, 2) * sqdists.get(1, 2) +
	     3 * sqdists.get(1, 2) * sqdists.get(1, 2))) /
      (8 * pow(sqdists.get(0, 1) * sqdists.get(0, 2) *
	       sqdists.get(1, 2), 2.5));
    *neg = AXILROD_TELLER_COEFF * (*neg);
    *pos = AXILROD_TELLER_COEFF * (*pos);
  }
  
  void EvalMinMax(double *negmin, double *negmax,
		  double *posmin, double *posmax) const {
    
    *negmin = -0.375 * 
      (distmat_.get(1, 0) * distmat_.get(1, 0) * distmat_.get(1, 0) + 
       distmat_.get(2, 0) * distmat_.get(2, 0) * distmat_.get(2, 0) +
       distmat_.get(2, 1) * distmat_.get(2, 1) * distmat_.get(2, 1)) /
      pow(distmat_.get(0, 1) * distmat_.get(0, 2) * distmat_.get(1, 2), 2.5);
    *posmin = (3 * distmat_.get(0, 1) * distmat_.get(0, 1) *
	       (distmat_.get(0, 2) + distmat_.get(1, 2)) +
	       3 * distmat_.get(0, 2) * distmat_.get(1, 2) *
	       (distmat_.get(0, 2) + distmat_.get(1, 2)) +
	       distmat_.get(0, 1) *
	       (3 * distmat_.get(0, 2) * distmat_.get(0, 2) +
		2 * distmat_.get(0, 2) * distmat_.get(1, 2) +
		3 * distmat_.get(1, 2) * distmat_.get(1, 2))) /
      (8 * pow(distmat_.get(1, 0) * distmat_.get(2, 0) *
	       distmat_.get(2, 1), 2.5));
    *negmax = -0.375 * 
      (distmat_.get(0, 1) * distmat_.get(0, 1) * distmat_.get(0, 1) + 
       distmat_.get(0, 2) * distmat_.get(0, 2) * distmat_.get(0, 2) +
       distmat_.get(1, 2) * distmat_.get(1, 2) * distmat_.get(1, 2)) /
      pow(distmat_.get(1, 0) * distmat_.get(2, 0) * distmat_.get(2, 1), 2.5);
    *posmax = (3 * distmat_.get(1, 0) * distmat_.get(1, 0) *
	       (distmat_.get(2, 0) + distmat_.get(2, 1)) +
	       3 * distmat_.get(2, 0) * distmat_.get(2, 1) *
	       (distmat_.get(2, 0) + distmat_.get(2, 1)) +
	       distmat_.get(1, 0) *
	       (3 * distmat_.get(2, 0) * distmat_.get(2, 0) +
		2 * distmat_.get(2, 0) * distmat_.get(2, 1) +
		3 * distmat_.get(2, 1) * distmat_.get(2, 1))) /
      (8 * pow(distmat_.get(0, 1) * distmat_.get(0, 2) *
	       distmat_.get(1, 2), 2.5));
    
    *negmin = AXILROD_TELLER_COEFF * (*negmin);
    *negmax = AXILROD_TELLER_COEFF * (*negmax);
    *posmin = AXILROD_TELLER_COEFF * (*posmin);
    *posmax = AXILROD_TELLER_COEFF * (*posmax);
  }

  const Matrix &EvalMinMaxDsqds
    (const ArrayList<DHrectBound<2> *> &node_bounds) {

    int num_nodes = node_bounds.size();

    for(index_t i = 0; i < num_nodes - 1; i++) {
      DHrectBound<2> *node_i_bound = node_bounds[i];

      for(index_t j = i + 1; j < num_nodes; j++) {
        DHrectBound<2> *node_j_bound = node_bounds[j];
        double dmin = node_i_bound->MinDistanceSq(*node_j_bound);
        double dmax = node_i_bound->MaxDistanceSq(*node_j_bound);

        distmat_.set(i, j, dmin);
        distmat_.set(j, i, dmax);
      }
    }
    
    return distmat_;
  }

  void Eval(const Matrix &data, const ArrayList<int> &indices, double *neg,
	    double *pos) {
    
    for(index_t i = 0; i < indices.size(); i++) {
      const double *i_col = data.GetColumnPtr(indices[i]);
      for(index_t j = i + 1; j < indices.size(); j++) {
	const double *j_col = data.GetColumnPtr(indices[j]);
	distmat_.set(i, j, la::DistanceSqEuclidean(data.n_rows(), i_col,
						   j_col));
      }
    }
    
    EvalUnnormOnSq(distmat_, neg, pos);
  }

  void EvalNodes(const ArrayList<DHrectBound<2> *> &node_bounds, 
		 double *negmin, double *negmax, double *posmin, 
		 double *posmax) {
    
    int num_nodes = node_bounds.size();

    for(index_t i = 0; i < num_nodes - 1; i++) {
      DHrectBound<2> *node_i_bound = node_bounds[i];

      for(index_t j = i + 1; j < num_nodes; j++) {
        DHrectBound<2> *node_j_bound = node_bounds[j];
        double dmin = node_i_bound->MinDistanceSq(*node_j_bound);
        double dmax = node_i_bound->MaxDistanceSq(*node_j_bound);

        distmat_.set(i, j, dmin);
        distmat_.set(j, i, dmax);
      }
    }
    
    EvalMinMax(negmin, negmax, posmin, posmax);
  }

};

#endif

#ifndef MULTIBODY_KERNEL_H
#define MULTIBODY_KERNEL_H

#include "fastlib/fastlib_int.h"
#include "u/dongryel/series_expansion/kernel_derivative.h"

class GaussianThreeBodyKernel {
  
 private:
  GaussianKernel kernel_;
  
  Matrix distmat_;

 public:

  GaussianThreeBodyKernel() {}
  
  ~GaussianThreeBodyKernel() {}

  // getters and setters
  double bandwidth_sq() const { return kernel_.bandwidth_sq(); }

  void Init(double bandwidth_in) {
    kernel_.Init(bandwidth_in);
    distmat_.Init(3, 3);
  }
  
  int order() {
    return 3;
  }

  double EvalUnnormOnSq(const Matrix &sqdists) const {
    
    double result = 1;

    for(index_t i = 0; i < sqdists.n_cols(); i++) {
      for(index_t j = i + 1; j < sqdists.n_cols(); j++) {

	result *= kernel_.EvalUnnormOnSq(sqdists.get(i, j));
      }
    }
    result *= 1e-27;
    return result;
  }

  void EvalMinMax(int order, double *min, double *max) const {

    *min = 1.0;
    *max = 1.0;

    for(index_t i = 0; i < order; i++) {
      for(index_t j = i + 1; j < order; j++) {
	*min = (*min) * kernel_.EvalUnnormOnSq(distmat_.get(j, i));
	*max = (*max) * kernel_.EvalUnnormOnSq(distmat_.get(i, j));
      }
    }
    *min = (*min) * 1e-27;
    *max = (*max) * 1e-27;
  }

  double Eval(const Matrix &data, const ArrayList<int> &indices) {
    
    for(index_t i = 0; i < indices.size(); i++) {
      const double *i_col = data.GetColumnPtr(indices[i]);
      for(index_t j = i + 1; j < indices.size(); j++) {
	const double *j_col = data.GetColumnPtr(indices[j]);
	distmat_.set(i, j, la::DistanceSqEuclidean(data.n_rows(), i_col,
						   j_col));
      }
    }
    
    return EvalUnnormOnSq(distmat_);
  }

  void EvalNodes(const ArrayList<DHrectBound<2> *> &node_bounds, double *min, 
		 double *max) {
    
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
    
    EvalMinMax(num_nodes, min, max);
  }

};

#endif

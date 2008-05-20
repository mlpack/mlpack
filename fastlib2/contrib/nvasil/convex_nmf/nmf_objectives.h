#ifndef NMF_OBJECTIVES_H_
#define NMF_OBJECTIVES_H_
#include "fastlib/fastlib.h"
#include "../l_bfgs/optimization_utils.h"

class BigSdpNmfObjective {
 public:
	void Init(fx_module *module, 
			ArrayList<index_t> &rows,
			ArrayList<index_t> &columns,
      ArrayList<double>  &values);
  void Destruct();
  void ComputeGradient(Matrix &coordinates, Matrix *gradient);
  void ComputeObjective(Matrix &coordinates, double *objective);
  void ComputeFeasibilityError(Matrix &coordinates, double *error);
  double ComputeLagrangian(Matrix &coordinates);
  void UpdateLagrangeMult(Matrix &coordinates);
  void Project(Matrix *coordinates);
  void set_sigma(double sigma); 
  void GiveInitMatrix(Matrix *init_data);
	bool IsDiverging(double objective); 
 
 private:
  fx_module *module_;
  double sigma_;
	index_t num_of_columns_;
	index_t num_of_rows_;
	index_t new_dim_;
	index_t rank_;
	index_t offset_h_;
	index_t offset_h_mat_; // we need this to know wher h matrix starts 
	ArrayList<index_t> rows_;
	ArrayList<index_t> columns_;
  ArrayList<double>  values_;
  Vector eq_lagrange_mult_; 

};

#include "nmf_objectives_impl.h"
#endif 

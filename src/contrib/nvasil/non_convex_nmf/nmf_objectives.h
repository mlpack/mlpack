#ifndef NMF_OBJECTIVES_H_
#define NMF_OBJECTIVES_H_
#include "fastlib/fastlib.h"
#include "../l_bfgs/optimization_utils.h"
#include "mlpack/allknn/allknn.h"
#include "contrib/nvasil/allkfn/allkfn.h"
#include "../mvu/mvu_objectives.h"


class BigSdpNmfObjectiveMaxVar {
 public:
	void Init(fx_module *module, 
			ArrayList<size_t> &rows,
			ArrayList<size_t> &columns,
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
  bool IsOptimizationOver(Matrix &coordinates, Matrix &gradient, double step);

 private:
  fx_module *module_;
  double sigma_;
	size_t num_of_columns_;
	size_t num_of_rows_;
	size_t new_dim_;
	size_t rank_;
	size_t offset_h_;
	size_t offset_h_mat_; // we need this to know where h matrix starts 
	ArrayList<size_t> rows_;
	ArrayList<size_t> columns_;
  ArrayList<double>  values_;
  Vector eq_lagrange_mult_; 
};

class BigSdpNmfObjectiveMinVar {
 public:
	void Init(fx_module *module, 
			ArrayList<size_t> &rows,
			ArrayList<size_t> &columns,
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
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);

 private:
  fx_module *module_;
  double sigma_;
	size_t num_of_columns_;
	size_t num_of_rows_;
	size_t new_dim_;
	size_t rank_;
	size_t offset_h_;
	size_t offset_h_mat_; // we need this to know wher h matrix starts 
	ArrayList<size_t> rows_;
	ArrayList<size_t> columns_;
  ArrayList<double>  values_;
  Vector eq_lagrange_mult_; 
};


class BigSdpNmfObjectiveMinVarIneq {
 public:
	void Init(fx_module *module, 
			ArrayList<size_t> &rows,
			ArrayList<size_t> &columns,
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
  bool IsOptimizationOver(Matrix &coordinates, Matrix &coordinates, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);

 private:
  fx_module *module_;
  double sigma_;
	size_t num_of_columns_;
	size_t num_of_rows_;
	size_t new_dim_;
	size_t rank_;
	size_t offset_h_;
	size_t offset_h_mat_; // we need this to know wher h matrix starts 
	ArrayList<size_t> rows_;
	ArrayList<size_t> columns_;
  ArrayList<double>  values_;
  Vector ineq_lagrange_mult_; 
};

class BigSdpNmfObjectiveMinVarDiagonalDominance {
 public:
	void Init(fx_module *module, 
			ArrayList<size_t> &rows,
			ArrayList<size_t> &columns,
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
  bool IsOptimizationOver(Matrix &coordinates, Matrix &coordinates, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);

 private:
  fx_module *module_;
  double sigma_;
  double sigma2_; // for inequalities
	size_t num_of_columns_;
	size_t num_of_rows_;
	size_t new_dim_;
	size_t offset_h_;
	size_t offset_h_mat_; // we need this to know wher h matrix starts 
	ArrayList<size_t> rows_;
	ArrayList<size_t> columns_;
  ArrayList<double>  values_;
  // this is for the equality for the values of the V
  Vector eq_lagrange_mult_;
  // Inequality for the diagonal dominance
  Vector ineq_lagrange_mult_; 
  Vector ineq_lagrange_mult1_;
};

class BigSdpNmfObjectiveMaxVarIsometric {
 public:
  static const size_t MAX_KNNS=30;
	void Init(fx_module *module, 
			ArrayList<size_t> &rows,
			ArrayList<size_t> &columns,
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
  bool IsOptimizationOver(Matrix &coordinates, Matrix &coordinates, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);

 private:
  fx_module *module_;
  double sigma_;
  double sigma2_; // for inequalities
	size_t num_of_columns_;
	size_t num_of_rows_;
	size_t new_dimension_;
  double grad_tolerance_;
	double desired_feasibility_error_;
  size_t h_offset_; // we need this to know wher h matrix starts 
  size_t w_offset_;
	ArrayList<size_t> rows_;
	ArrayList<size_t> columns_;
  ArrayList<double>  values_;
  AllkNN allknn_;
  size_t knns_;
  size_t leaf_size_;
  ArrayList<std::pair<size_t, size_t> > nearest_neighbor_pairs_;
  ArrayList<double> nearest_distances_;
  size_t num_of_nearest_pairs_;
  double sum_of_furthest_distances_;
  double v_norm_;
  double sum_all_distances_;
  double sigma_ratio_;
  double infeasibility1_;
  double previous_infeasibility1_;
  double infeasibility2_;
  // this is for the equality for the values of the V
  Vector eq_lagrange_mult1_;
  Vector eq_lagrange_mult2_;
};

class BigSdpNmfObjectiveMaxVarIsometricRemoveMean {
 public:
  static const size_t MAX_KNNS=30;
	void Init(fx_module *module, 
			ArrayList<size_t> &rows,
			ArrayList<size_t> &columns,
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
  bool IsOptimizationOver(Matrix &coordinates, Matrix &coordinates, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);

 private:
  fx_module *module_;
  double sigma_;
  double sigma2_; // for inequalities
	size_t num_of_columns_;
	size_t num_of_rows_;
	size_t new_dimension_;
  double grad_tolerance_;
	double desired_feasibility_error_;
  size_t h_offset_; // we need this to know wher h matrix starts 
  size_t w_offset_;
	ArrayList<size_t> rows_;
	ArrayList<size_t> columns_;
  ArrayList<double>  values_;
  AllkNN allknn_;
  size_t knns_;
  size_t leaf_size_;
  ArrayList<std::pair<size_t, size_t> > nearest_neighbor_pairs_;
  ArrayList<double> nearest_distances_;
  size_t num_of_nearest_pairs_;
  double sum_of_furthest_distances_;
  double v_norm_;
  double sum_all_distances_;
  double sigma_ratio_;
  double infeasibility1_;
  double infeasibility2_;
  // this is for the equality for the values of the V
  Vector eq_lagrange_mult1_;
  Vector eq_lagrange_mult2_;
};

class BigSdpNmfObjectiveMaxFurthestIsometric {
 public:
  static const size_t MAX_KNNS=30;
	void Init(fx_module *module, 
			ArrayList<size_t> &rows,
			ArrayList<size_t> &columns,
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
  bool IsOptimizationOver(Matrix &coordinates, Matrix &coordinates, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);

 private:
  fx_module *module_;
  double sigma_;
  double sigma2_; // for inequalities
	size_t num_of_columns_;
	size_t num_of_rows_;
	size_t new_dimension_;
  double grad_tolerance_;
	double desired_feasibility_error_;
  size_t h_offset_; // we need this to know wher h matrix starts 
  size_t w_offset_;
	ArrayList<size_t> rows_;
	ArrayList<size_t> columns_;
  ArrayList<double>  values_;
  AllkNN allknn_;
  size_t knns_;
  AllkFN allkfn_;
  size_t leaf_size_;
  ArrayList<std::pair<size_t, size_t> > nearest_neighbor_pairs_;
  ArrayList<double> nearest_distances_;
  size_t num_of_nearest_pairs_;
  size_t num_of_furthest_pairs_;
  ArrayList<std::pair<size_t, size_t> > furthest_neighbor_pairs_;
  ArrayList<double> furthest_distances_;
  double sum_of_furthest_distances_;
  double v_norm_;
  double sum_all_distances_;
  double sigma_ratio_;
  double infeasibility1_;
  double previous_infeasibility1_;
  double infeasibility2_;
  // this is for the equality for the values of the V
  Vector eq_lagrange_mult1_;
  Vector eq_lagrange_mult2_;
};

class NmfObjectiveIsometric {
 public:
  static const size_t MAX_KNNS=30;
	void Init(fx_module *module, 
			ArrayList<size_t> &rows,
			ArrayList<size_t> &columns,
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
  bool IsOptimizationOver(Matrix &coordinates, Matrix &coordinates, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);

 private:
  fx_module *module_;
  double sigma_;
	size_t num_of_columns_;
	size_t num_of_rows_;
	size_t new_dimension_;
  double grad_tolerance_;
	double desired_feasibility_error_;
  size_t h_offset_; // we need this to know wher h matrix starts 
  size_t w_offset_;
	ArrayList<size_t> rows_;
	ArrayList<size_t> columns_;
  ArrayList<double>  values_;
  AllkNN allknn_;
  size_t knns_;
  size_t leaf_size_;
  ArrayList<std::pair<size_t, size_t> > nearest_neighbor_pairs_;
  ArrayList<double> nearest_distances_;
  size_t num_of_nearest_pairs_;
  double sum_of_furthest_distances_;
  double v_norm_;
  double sum_all_distances_;
  double sigma_ratio_;
  double infeasibility2_;
  double previous_infeasibility2_;
  // this is for the equality for the values of the V
  Vector eq_lagrange_mult2_;
};


class BigSdpNmfObjectiveMinVarLocal {
 public:
	void Init(fx_module *module, 
			ArrayList<size_t> &rows,
			ArrayList<size_t> &columns,
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
  bool IsOptimizationOver(Matrix &coordinates, Matrix &coordinates, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);

 private:
  fx_module *module_;
  double sigma_;
	size_t num_of_columns_;
	size_t num_of_rows_;
	size_t new_dim_;
	size_t rank_;
	size_t offset_h_;
	size_t offset_h_mat_; // we need this to know wher h matrix starts 
	ArrayList<size_t> rows_;
	ArrayList<size_t> columns_;
  ArrayList<double>  values_;
  // this is for the equality for the values of the V
  Vector eq_lagrange_mult_;
  // Inequality for the diagonal dominance
  Vector ineq_lagrange_mult_; 
};


class  ClassicNmfObjective {
 public:
	void Init(fx_module *module, 
			ArrayList<size_t> &rows,
			ArrayList<size_t> &columns,
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
  bool IsOptimizationOver(Matrix &coordinates, Matrix &coordinates, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &coordinates, double step);

 private:
  fx_module *module_;
  double sigma_;
	size_t num_of_columns_;
	size_t num_of_rows_;
	size_t new_dimension_;
  double grad_tolerance_;
	size_t h_offset_;
  size_t w_offset_;
	size_t offset_h_mat_; // we need this to know wher h matrix starts 
	ArrayList<size_t> rows_;
	ArrayList<size_t> columns_;
  ArrayList<double>  values_;
};


#include "nmf_objectives_impl.h"
#endif 

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
  bool IsOptimizationOver(Matrix &coordinates, Matrix &gradient, double step);

 private:
  fx_module *module_;
  double sigma_;
	index_t num_of_columns_;
	index_t num_of_rows_;
	index_t new_dim_;
	index_t rank_;
	index_t offset_h_;
	index_t offset_h_mat_; // we need this to know where h matrix starts 
	ArrayList<index_t> rows_;
	ArrayList<index_t> columns_;
  ArrayList<double>  values_;
  Vector eq_lagrange_mult_; 
};

class BigSdpNmfObjectiveMinVar {
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
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);

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


class BigSdpNmfObjectiveMinVarIneq {
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
  bool IsOptimizationOver(Matrix &coordinates, Matrix &coordinates, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);

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
  Vector ineq_lagrange_mult_; 
};

class BigSdpNmfObjectiveMinVarDiagonalDominance {
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
  bool IsOptimizationOver(Matrix &coordinates, Matrix &coordinates, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);

 private:
  fx_module *module_;
  double sigma_;
  double sigma2_; // for inequalities
	index_t num_of_columns_;
	index_t num_of_rows_;
	index_t new_dim_;
	index_t offset_h_;
	index_t offset_h_mat_; // we need this to know wher h matrix starts 
	ArrayList<index_t> rows_;
	ArrayList<index_t> columns_;
  ArrayList<double>  values_;
  // this is for the equality for the values of the V
  Vector eq_lagrange_mult_;
  // Inequality for the diagonal dominance
  Vector ineq_lagrange_mult_; 
  Vector ineq_lagrange_mult1_;
};

class BigSdpNmfObjectiveMaxVarIsometric {
 public:
  static const index_t MAX_KNNS=30;
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
  bool IsOptimizationOver(Matrix &coordinates, Matrix &coordinates, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);

 private:
  fx_module *module_;
  double sigma_;
  double sigma2_; // for inequalities
	index_t num_of_columns_;
	index_t num_of_rows_;
	index_t new_dimension_;
  double grad_tolerance_;
	double desired_feasibility_error_;
  index_t h_offset_; // we need this to know wher h matrix starts 
  index_t w_offset_;
	ArrayList<index_t> rows_;
	ArrayList<index_t> columns_;
  ArrayList<double>  values_;
  AllkNN allknn_;
  index_t knns_;
  index_t leaf_size_;
  ArrayList<std::pair<index_t, index_t> > nearest_neighbor_pairs_;
  ArrayList<double> nearest_distances_;
  index_t num_of_nearest_pairs_;
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
  static const index_t MAX_KNNS=30;
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
  bool IsOptimizationOver(Matrix &coordinates, Matrix &coordinates, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);

 private:
  fx_module *module_;
  double sigma_;
  double sigma2_; // for inequalities
	index_t num_of_columns_;
	index_t num_of_rows_;
	index_t new_dimension_;
  double grad_tolerance_;
	double desired_feasibility_error_;
  index_t h_offset_; // we need this to know wher h matrix starts 
  index_t w_offset_;
	ArrayList<index_t> rows_;
	ArrayList<index_t> columns_;
  ArrayList<double>  values_;
  AllkNN allknn_;
  index_t knns_;
  index_t leaf_size_;
  ArrayList<std::pair<index_t, index_t> > nearest_neighbor_pairs_;
  ArrayList<double> nearest_distances_;
  index_t num_of_nearest_pairs_;
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
  static const index_t MAX_KNNS=30;
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
  bool IsOptimizationOver(Matrix &coordinates, Matrix &coordinates, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);

 private:
  fx_module *module_;
  double sigma_;
  double sigma2_; // for inequalities
	index_t num_of_columns_;
	index_t num_of_rows_;
	index_t new_dimension_;
  double grad_tolerance_;
	double desired_feasibility_error_;
  index_t h_offset_; // we need this to know wher h matrix starts 
  index_t w_offset_;
	ArrayList<index_t> rows_;
	ArrayList<index_t> columns_;
  ArrayList<double>  values_;
  AllkNN allknn_;
  index_t knns_;
  AllkFN allkfn_;
  index_t leaf_size_;
  ArrayList<std::pair<index_t, index_t> > nearest_neighbor_pairs_;
  ArrayList<double> nearest_distances_;
  index_t num_of_nearest_pairs_;
  index_t num_of_furthest_pairs_;
  ArrayList<std::pair<index_t, index_t> > furthest_neighbor_pairs_;
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
  static const index_t MAX_KNNS=30;
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
  bool IsOptimizationOver(Matrix &coordinates, Matrix &coordinates, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);

 private:
  fx_module *module_;
  double sigma_;
	index_t num_of_columns_;
	index_t num_of_rows_;
	index_t new_dimension_;
  double grad_tolerance_;
	double desired_feasibility_error_;
  index_t h_offset_; // we need this to know wher h matrix starts 
  index_t w_offset_;
	ArrayList<index_t> rows_;
	ArrayList<index_t> columns_;
  ArrayList<double>  values_;
  AllkNN allknn_;
  index_t knns_;
  index_t leaf_size_;
  ArrayList<std::pair<index_t, index_t> > nearest_neighbor_pairs_;
  ArrayList<double> nearest_distances_;
  index_t num_of_nearest_pairs_;
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
  bool IsOptimizationOver(Matrix &coordinates, Matrix &coordinates, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &gradient, double step);

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
  // this is for the equality for the values of the V
  Vector eq_lagrange_mult_;
  // Inequality for the diagonal dominance
  Vector ineq_lagrange_mult_; 
};


class  ClassicNmfObjective {
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
  bool IsOptimizationOver(Matrix &coordinates, Matrix &coordinates, double step);
  bool IsIntermediateStepOver(Matrix &coordinates, Matrix &coordinates, double step);

 private:
  fx_module *module_;
  double sigma_;
	index_t num_of_columns_;
	index_t num_of_rows_;
	index_t new_dimension_;
  double grad_tolerance_;
	index_t h_offset_;
  index_t w_offset_;
	index_t offset_h_mat_; // we need this to know wher h matrix starts 
	ArrayList<index_t> rows_;
	ArrayList<index_t> columns_;
  ArrayList<double>  values_;
};


#include "nmf_objectives_impl.h"
#endif 

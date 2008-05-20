#ifndef NMF_ENGINE_H_
#define NMF_ENGINE_H_
#include "fastlib/fastlib.h"
#include "../l_bfgs/l_bfgs.h"
#include "nmf_objectives.h"

class NmfEngine {
 public:
	 void  Init(fx_module *module) {
 		 module_=module;
	   std::string data_file=fx_param_str_req(module_, "data_file");
     sdp_rank_=fx_param_int(module_,  "sdp_rank", 5);
		 new_dim_=fx_param_int(module_, "new_dim", 3);
		 Matrix data_mat;
		 data::Load(data_file.c_str(), &data_mat);
		 PreprocessData(data_mat);
		 fx_module *opt_function_module=fx_submodule(module_, "optfun");
     fx_set_param_int(opt_function_module, "rank", sdp_rank_);
		 fx_set_param_int(opt_function_module, "new_dim", new_dim_);
		 opt_function_.Init(opt_function_module, rows_, columns_, values_);
		 fx_module *l_bfgs_module=fx_submodule(module_, "l_bfgs");
     Matrix init_data;
		 opt_function_.GiveInitMatrix(&init_data);
	   fx_set_param_int(l_bfgs_module, "num_of_points", init_data.n_cols());
		 fx_set_param_int(l_bfgs_module, "new_dimension", init_data.n_rows());
     engine_.Init(&opt_function_, l_bfgs_module);
	 	 engine_.set_coordinates(init_data);
	 }
	 void Destruct() {
	   
	 };
	 void ComputeNmf() {
		  Matrix init_data;
			opt_function_.GiveInitMatrix(&init_data);
		  engine_.set_coordinates(init_data);
	    engine_.ComputeLocalOptimumBFGS();
      Matrix result;
      engine_.GetResults(&result);
			w_mat_.Init(num_of_rows_, new_dim_);
			h_mat_.Init(new_dim_, num_of_columns_);
			for(index_t i=0; i<num_of_rows_; i++) {
			  for(index_t j=0; j<new_dim_; j++) {
				  w_mat_.set(i, j, result.get(0, i*new_dim_+j));
				}
			}
			index_t offset_h=num_of_rows_*new_dim_;
			for(index_t i=0; i<num_of_columns_; i++) {
			  for(index_t j=0; j<new_dim_; j++) {
				   h_mat_.set(j, i , result.get(0, offset_h+i*new_dim_+j ));
				}
			}
	 }
	 void GetW(Matrix *w_mat) {
	   w_mat->Copy(w_mat_);
	 }
	 void GetH(Matrix *h_mat) {
	   h_mat->Copy(h_mat_);
	 }
	 
 private:
	fx_module *module_;
  LBfgs<BigSdpNmfObjective> engine_;
	BigSdpNmfObjective opt_function_;
	ArrayList<index_t> rows_;
	ArrayList<index_t> columns_;
	ArrayList<double> values_;
	index_t new_dim_;
	index_t sdp_rank_;
	Matrix w_mat_;
	Matrix h_mat_;
	index_t num_of_rows_; // number of unique rows, otherwise the size of W
	index_t num_of_columns_; // number of unique columns, otherwise the size of H
 
	void PreprocessData(Matrix &data_mat) {
	  values_.Init();
		rows_.Init();
		columns_.Init();
		for(index_t i=0; i<data_mat.n_rows(); i++) {
		  for(index_t j=0; j< data_mat.n_cols(); j++) {
			  values_.PushBackCopy(data_mat.get(i, j));
				rows_.PushBackCopy(i);
				columns_.PushBackCopy(j);
			}
		}
	}
};

#endif

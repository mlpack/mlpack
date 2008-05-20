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
		 PreprcessData(data_mat);
     opt_function_.Init(fx_submodule(module,"optfun"), 
				 rows_, columns_, values_);
     engine.Init(&opt_function, l_bfgs_node);
	 }
	 void Destruct() {
	   
	 };
	 void ComputeNmf() {
		  Matrix init_data;
			opt_function_.GiveInitMatrix(&init_data);
		  engine.set_coordinates(init_data);
	    engine_.ComputeLocalOptimumBFGS();
      Matrix result;
      engine_.GetResults(&result);
			w_mat_.Init(num_of_rows_, new_dim_);
			h_mat_.Init(new_dim_, num_of_columns_);
			for(index_t i=0; i<num_of_rows_; i++) {
			  for(index_t j=0; j<new_dim_; j++) {
				  w_mat_.set_(i, j, results.get(0, i*new_dim_+j));
				}
			}
			index_t offset_h=num_of_rows_*new_dim_;
			for(index_t i=0; i<num_of_columns_; i++) {
			  for(index_t j=0; j<new_dim_; j++) {
				   h_mat.set(j, i , results(0, offset_h+i*new_dim_+j ));
				}
			}
	 }
	 void GetW(Matrix *w_mat) {
	   h_mat->Copy(*w_mat);
	 }
	 void GetH(Matrix *h_mat) {
	   w_mat->Copy(*h_mat);
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
			  values_.PushBack(values_.get(i, j));
				rows_.PushBack(i);
				columns_.PushBack(j);
			}
		}
	}
};

#endif

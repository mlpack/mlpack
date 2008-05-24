/*
 * =====================================================================================
 * 
 *       Filename:  sdm_nmf_engine.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  05/24/2008 12:52:57 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef SDP_NMF_ENGINE_H_
#define SDP_NMF_ENGINE_H_
#include "fastlib/fastlib.h"
#include "../l_bfgs/l_bfgs.h"
#include "sdp_objectives.h"

template<typename SdpNmfObjective>
class SdpNmfEngine {
 public:
	 void  Init(fx_module *module) {
 		 module_=module;
	   std::string data_file=fx_param_str_req(module_, "data_file");
		 new_dim_=fx_param_int(module_, "new_dim", 3);
		 Matrix data_mat;
		 if (data::Load(data_file.c_str(), &data_mat)==SUCCESS_FAIL) {
       FATAL("Terminating...");
     }
     NOTIFY("Factoring a %i x %i matrix in %i x %i and %i x %i\n",
         data_mat.n_rows(), data_mat.n_cols(), data_mat.n_rows(), new_dim_,
         new_dim_, data_mat.n_cols());
		 PreprocessData(data_mat);
		 fx_module *opt_function_module=fx_submodule(module_, "optfun");
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
     engine_.ComputeLocalOptimumBFGS();
     Matrix result;
     engine_.GetResults(&result);
		 w_mat_.Init(num_of_rows_, new_dim_);
		 h_mat_.Init(new_dim_, num_of_columns_);
     w_mat_.CopyColumnFromMat(0, num_of_rows_, result); 
     h_mat_.CopyColumnFromMat(num_of_rows_, num_of_columns_, result);
     data::Save("result.csv", result);
     // now compute reconstruction error
     Matrix v_rec;
     la::MulInit(w_mat_, h_mat_, &v_rec);
     double error=0;
     double v_sum=0;
     for(index_t i=0; i<values_.size(); i++) {
       index_t r=rows_[i];
       index_t c=columns_[i];
       error+=fabs(v_rec.get(r, c)-values_[i]);
       v_sum+=values_[i];
     }
     NOTIFY("Reconstruction error: %lg%%\n", error*100/v_sum);
	 }
   
	 void GetW(Matrix *w_mat) {
	   w_mat->Copy(w_mat_);
	 }
	 void GetH(Matrix *h_mat) {
	   h_mat->Copy(h_mat_);
	 }
	 
 private:
	fx_module *module_;
  LBfgs<SdpNmfObjective> engine_;
	SdpNmfObjective  opt_function_;
	ArrayList<index_t> rows_;
	ArrayList<index_t> columns_;
	ArrayList<double> values_;
	index_t new_dim_;
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
    num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
    num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
	}
};

#endif

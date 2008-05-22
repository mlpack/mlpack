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
		 if (data::Load(data_file.c_str(), &data_mat)==SUCCESS_FAIL) {
       FATAL("Terminating...");
     }
     NOTIFY("Factoring a %i x %i matrix in %i x %i and %i x %i\n",
         data_mat.n_rows(), data_mat.n_cols(), data_mat.n_rows(), new_dim_,
         new_dim_, data_mat.n_cols());
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
      Vector s;
      Matrix u_mat, vt_mat;
      success_t success=la::SVDInit(result, &s, &u_mat, &vt_mat);
      if (success==SUCCESS_PASS) {
        NOTIFY("Svd success full...\n");
        std::string temp;
        char buffer[64];
        for(index_t i=0; i<s.length(); i++) {
          sprintf(buffer, "%lg ", s[i]);
          temp.append(buffer);
        }
        NOTIFY("Singular values: %s", temp.c_str());
      } else {
        FATAL("Svd failed soething is wrong...\n");
      }
      bool negative_flag=false;
      for (index_t i=0; i<vt_mat.n_cols(); i++) {
        if (vt_mat.get(0, i)<0) {
          negative_flag=true;
        }
        if (negative_flag==true && vt_mat.get(0, i)>0) {
          NONFATAL("Method failed, first eigenvector has positive and negative elements");
          break;          
        }
      } 
      if (negative_flag==true) {
        la::Scale(-1, &vt_mat);
      }
      opt_function_.Project(&vt_mat);
      for(index_t i=0; i<num_of_rows_; i++) {
			  for(index_t j=0; j<new_dim_; j++) {
				  w_mat_.set(i, j, s[0]*vt_mat.get(0, i*new_dim_+j));
				}
			}
			index_t offset_h=num_of_rows_*new_dim_;
			for(index_t i=0; i<num_of_columns_; i++) {
			  for(index_t j=0; j<new_dim_; j++) {
				   h_mat_.set(j, i , s[0]*vt_mat.get(0, offset_h+i*new_dim_+j ));
				}
			}
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
      data::Save("result.csv", result);
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
    num_of_rows_=*std::max_element(rows_.begin(), rows_.end())+1;
    num_of_columns_=*std::max_element(columns_.begin(), columns_.end())+1;
	}
};

#endif

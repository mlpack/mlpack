/*
 * =====================================================================================
 * 
 *       Filename:  geometric_nmf_seq_engine.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  07/08/2008 11:58:53 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef GEOMETRIC_NMF_SEQ_ENGINE_H_
#define GEOMETRIC_NMF_SEQ_ENGINE_H_

template<typename GeometricNmfObjective>
class GeometricNmfSeqEngine {
 public:
   GeometricNmfSeqEngine() {
     engine_=NULL;
   }
   ~GeometricNmfSeqEngine() {
     if (engine_!=NULL) {
       delete engine_;
     }
   }
	 void  Init(fx_module *module) {
 		 module_=module;
	   std::string data_file=fx_param_str_req(module_, "data_file");
		 new_dim_=fx_param_int(module_, "new_dim", 3);
		 Matrix data_mat;
		 if (data::Load(data_file.c_str(), &data_mat)==SUCCESS_FAIL) {
       FATAL("Terminating...");
     }
     PreprocessData(data_mat);
     NOTIFY("Factoring a %i x %i matrix in %i x %i and %i x %i\n",
         data_mat.n_rows(), data_mat.n_cols(), data_mat.n_rows(), new_dim_,
         new_dim_, data_mat.n_cols());
		 l_bfgs_module_=fx_submodule(module_, "l_bfgs");
	 }
	 void Destruct() {
     if (engine_!=NULL) {
	     delete engine_;
       engine_ = NULL;
     }
     GeometricNmfObjective  opt_function_;
	   rows_.Renew();
	   columns_.Renew();
	   values_.Renew();
	   w_mat_.Destruct();
	   h_mat_.Destruct();

	 };
	 void ComputeNmf() {
     Matrix result;
     result.Init(new_dim_, num_of_rows_+num_of_columns_);
     result.SetAll(0.0);
	   fx_set_param_int(l_bfgs_module_, "num_of_points",
         num_of_rows_+num_of_columns_);
		 fx_set_param_int(l_bfgs_module_, "new_dimension", 1);
     w_mat_.Init(new_dim_, num_of_rows_);
		 h_mat_.Init(new_dim_, num_of_columns_);
     ArrayList<double> values_original;
     values_original.InitCopy(values_, values_.capacity());
		 fx_module *opt_function_module=fx_submodule(module_, "optfun");
		 fx_set_param_int(opt_function_module, "new_dim", 1);
     
     for(index_t i=0; i<new_dim_; i++) {
       opt_function_= new GeometricNmfObjective();
		   opt_function_->Init(opt_function_module, rows_, 
           columns_, values_);
       engine_=new LBfgs<GeometricNmfObjective>();
       engine_->Init(opt_function_, l_bfgs_module_);
       Matrix rank_one_mat;
       opt_function_->GiveInitMatrix(&rank_one_mat);
       engine_->set_coordinates(rank_one_mat);
       rank_one_mat.Destruct();
       engine_->ComputeLocalOptimumBFGS();
       engine_->GetResults(&rank_one_mat);
       delete engine_;
       delete opt_function_;
  		 for(index_t j=0; j<num_of_rows_+num_of_columns_; j++) {
         result.set(i, j, exp(rank_one_mat.get(0, j)));
       }
       w_mat_.CopyColumnFromMat(0, 0, num_of_rows_, result); 
       h_mat_.CopyColumnFromMat(0, 
         num_of_rows_, num_of_columns_, result);
       // now compute reconstruction error
       Matrix v_rec;
       la::MulTransAInit(w_mat_, h_mat_, &v_rec);
       double error=0;
       double v_sum=0;
       for(index_t i=0; i<values_.size(); i++) {
         index_t r=rows_[i];
         index_t c=columns_[i];
         error+=values_original[i]-v_rec.get(r, c);
         values_[i]=values_original[i]-v_rec.get(r, c);
         v_sum+=values_original[i];
       }
       NOTIFY("********Reconstruction error: %lg%%\n", error*100/v_sum);
       engine_=NULL;
     }
     data::Save("result.csv", result);
   } 
	 void GetW(Matrix *w_mat) {
	   w_mat->Copy(w_mat_);
	 }
	 void GetH(Matrix *h_mat) {
	   h_mat->Copy(h_mat_);
	 }
	 
 private:
	fx_module *module_;
  fx_module *l_bfgs_module_;
  LBfgs<GeometricNmfObjective> *engine_;
	GeometricNmfObjective  *opt_function_;
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



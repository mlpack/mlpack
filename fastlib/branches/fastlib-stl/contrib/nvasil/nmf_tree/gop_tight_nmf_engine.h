/*
 * =====================================================================================
 * 
 *       Filename:  gop_tight_nmf_engine.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  08/19/2008 04:11:39 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */
#ifndef GOP_TIGHT_NMF_ENGINE_H_
#define GOP_TIGHT_NMF_ENGINE_H_

#include "fastlib/fastlib.h"
#include "../l_bfgs/l_bfgs.h"
#include "../convex_nmf/gop_nmf.h"
#include "../non_convex_nmf/nmf_objectives.h"
#include "relaxed_nmf_bound_tightener.h"

class GopTightNmfEngine {
 public:
  void Init(fx_module *module, Matrix &data_matrix);
  void Destruct();
  void TightenBounds();
  void ComputeGlobalOptimum();
  
 private:
  fx_module *module_;
  RelaxedNmfIsometric relaxed_nmf_;
  RelaxedNmfIsometricBoxTightener relaxed_nmf_box_tightener_;
  ClassicNmfObjective classic_nmf_objective_;
  LBfgs<RelaxedNmfIsometric>  relaxed_nmf_optimizer_;
  LBfgs<RelaxedNmfIsometricBoxTightener> box_tightener_optimizer_;
  LBfgs<ClassicNmfObjective> classic_nmf_optimizer_;
  GopNmfEngine gop_nmf_engine_;
  Matrix *current_solution_;
  ArrayList<index_t> rows_;
  ArrayList<index_t> columns_;
  ArrayList<double> values_;
  index_t num_of_rows_;
  index_t num_of_columns_;
  index_t new_dimension_;
  Vector lower_box_;
  Vector upper_box_;
  double  objective_minimum_upper_bound_;
};

#include "gop_tight_nmf_engine_impl.h"
#endif

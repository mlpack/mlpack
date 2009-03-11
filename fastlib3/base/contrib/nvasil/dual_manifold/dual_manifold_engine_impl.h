/*
 * =====================================================================================
 * 
 *       Filename:  dual_manifold_engine_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  03/18/2008 11:34:02 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

template<typename OptimizedFunction>
void DualManifoldEngine<OptimizedFunction>::Init(datanode *module, 
    // index pairs to consider from the matrix (row,column) pairs
    ArrayList<std::pair<index_t, index_t> > &pairs_to_consider, 
    // The values of the (row, column) values, also known as the dot products
    ArrayList<double> &dot_prod_values) {
  
  // From this matrix we deduce the size of the matrix D we are trying to decompose
  // We want to find out how many rows and columns he has
  index_t num_of_rows=0;
  index_t num_of_cols=0;
  for (index_t i=0; i<pairs_to_consider.size(); i++) {
    if (pairs_to_consider[i].first > num_of_rows) {
      num_of_rows=pairs_to_consider[i].first;
    }
    if (pairs_to_consider[i].second > num_of_cols) {
      num_of_cols=pairs_to_consider[i].second;
    }
  }
  num_of_rows+=1;
  num_of_cols+=1;
  module_=module;
  datanode *l_bfgs_node1=fx_submodule(module_, "/l_bfgs", "l_bfgs1");
  datanode *l_bfgs_node2=fx_submodule(module_, "/l_bfgs", "l_bfgs2");

  max_iterations_=fx_param_int(module_, "max_iterations", 100000);
  desired_feasibility_=fx_param_double(module_, "desired_feasibility",  1000);
  feasibility_tolerance_ = fx_param_double(module, "feasibility_tolerance", 0.001); 
  num_of_components_=fx_param_int(module_, "components", 40);
    //we need to insert the number of points
  char buffer[128];
  sprintf(buffer, "%i", num_of_components_);

  fx_set_param(l_bfgs_node1, "new_dimension", buffer);
  fx_set_param(l_bfgs_node2, "new_dimension", buffer);
 
  sprintf(buffer, "%i", num_of_rows);
  fx_set_param(l_bfgs_node1, "num_of_points", buffer);

  sprintf(buffer, "%i", num_of_cols);
  fx_set_param(l_bfgs_node2, "num_of_points", buffer);

  l_bfgs1_.Init(&optimized_function1_, l_bfgs_node1);
  l_bfgs2_.Init(&optimized_function2_, l_bfgs_node2);

  optimized_function1_.Init(fx_param_node(module_, "opt1"), 
      l_bfgs2_.coordinates(), 
      pairs_to_consider,
      dot_prod_values);

  for(index_t i=0; i<pairs_to_consider.size(); i++) {
    std::swap(pairs_to_consider[i].first, pairs_to_consider[i].second);
  }
  optimized_function2_.Init(fx_param_node(module_, "opt2"), 
      l_bfgs1_.coordinates(), 
      pairs_to_consider,
      dot_prod_values); 
}

template<typename OptimizedFunction>
void DualManifoldEngine<OptimizedFunction>::Destruct() {
  l_bfgs1_.Destruct();
  l_bfgs2_.Destruct();
}

template<typename OptimizedFunction>
void DualManifoldEngine<OptimizedFunction>::ComputeLocalOptimum() {
  //l_bfgs1_.set_max_iterations(10);
  //l_bfgs2_.set_max_iterations(10);
  double old_feasibility_error=DBL_MAX;
  optimized_function1_.ComputeFeasibilityError(*l_bfgs1_.coordinates(), 
      &old_feasibility_error);
  double feasibility_error=DBL_MAX;
  for(index_t i=0; i<max_iterations_; i++) {
    NOTIFY("Optimizing for W...");
    l_bfgs1_.ComputeLocalOptimumBFGS();
    optimized_function1_.ComputeFeasibilityError(*l_bfgs1_.coordinates(), &feasibility_error);
    if (feasibility_error<desired_feasibility_ || 
        fabs(feasibility_error-old_feasibility_error)/old_feasibility_error 
        < feasibility_tolerance_) {
      break;
    }
    NOTIFY("Optimizing for H...");
    l_bfgs2_.ComputeLocalOptimumBFGS();
    optimized_function2_.ComputeFeasibilityError(*l_bfgs2_.coordinates(), &feasibility_error);
    if (feasibility_error<desired_feasibility_ || 
        fabs(feasibility_error-old_feasibility_error)/old_feasibility_error 
        < feasibility_tolerance_) {
      break;
    }
    old_feasibility_error=feasibility_error;
  }
  NOTIFY("feasibility_error:%lg relative error_change:%lg", feasibility_error,
       fabs(feasibility_error-old_feasibility_error)/(old_feasibility_error)); 
}

template<typename OptimizedFunction>
double DualManifoldEngine<OptimizedFunction>::ComputeEvaluationTest(
    ArrayList<std::pair<index_t, index_t> > &pairs_to_consider, 
    ArrayList<double> &dot_prod_values) {
 
  double error=0;
  Matrix *w_mat = l_bfgs1_.coordinates();
  Matrix *h_mat = l_bfgs2_.coordinates();
  for(index_t i=0; i<pairs_to_consider.size(); i++) {
    index_t ind1=pairs_to_consider[i].first;
    index_t ind2=pairs_to_consider[i].second;
    double eval=fabs(la::Dot(num_of_components_, 
        w_mat->GetColumnPtr(ind1),
        h_mat->GetColumnPtr(ind2)) - dot_prod_values[i]);
/*    if (eval<1) {
      eval=1;
    }
    if (eval>5) {
      eval=5;
    }
    eval=int(eval);
*/
    error+=eval;
  }
  
  error/=pairs_to_consider.size();
  fx_format_result(module_, "evaluation error", "%lg", error);
  return error;
}

template<typename OptimizedFunction>
double DualManifoldEngine<OptimizedFunction>::ComputeRandomEvaluationTest(
    ArrayList<std::pair<index_t, index_t> > &pairs_to_consider, 
    ArrayList<double> &dot_prod_values) {
 
  double error=0;
 for(index_t i=0; i<pairs_to_consider.size(); i++) {
   double eval=fabs(math::RandInt(1, 6)- dot_prod_values[i]);
/*    if (eval<1) {
      eval=1;
    }
    if (eval>5) {
      eval=5;
    }
    eval=int(eval);
*/
    error+=eval;
  }
  
  error/=pairs_to_consider.size();
  fx_format_result(module_, "random predictor error", "%lg", error);
  return error;
}


template<typename OptimizedFunction>
Matrix *DualManifoldEngine<OptimizedFunction>::Matrix1() {
  return l_bfgs1_.coordinates();
}

template<typename OptimizedFunction>
Matrix *DualManifoldEngine<OptimizedFunction>::Matrix2() {
  return l_bfgs2_.coordinates();
}



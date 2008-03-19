/*
 * =====================================================================================
 * 
 *       Filename:  dual_manifold_objective_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  03/18/2008 08:09:51 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

void DualMaxVarianc::eInit(datanode *module, Matrix *other_part, 
      ArrayList<std::pair<index_t, index_t> > *pairs_to_consider,
      ArrayList<double> *dot_prod_values);
void DualMaxVariance::ComputeGradient(Matrix &coordinates, Matrix *gradient);
void DualMaxVariance::ComputeObjective(Matrix &coordinates, double *objective);
void DualMaxVariance::ComputeFeasibilityError(Matrix &coordinates, double *error);
double DualMaxVariance::ComputeLagrangian(Matrix &coordinates);
void DualMaxVariance::UpdateLagrangeMult(Matrix &coordinates);
void DualMaxVariance::Project(Matrix *coordinates);
void DualMaxVariance::set_sigma(double sigma); 
 

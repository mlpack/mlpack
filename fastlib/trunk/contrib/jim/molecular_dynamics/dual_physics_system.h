/**
 * @file physics_system.h
 *
 * @author Jim Waters (jwaters6@gatech.edu)
 *
 *
 */


#ifndef DUAL_PHYSICS_SYSTEM_H
#define DUAL_PHYSICS_SYSTEM_H

#include "fastlib/fastlib.h"
#include "fastlib/fastlib_int.h"
#include "particle_tree.h"
#include "force_error.h"
#include "raddist.h"
#define PI 3.14159265358979


const fx_entry_doc param_entries[] = {
  {"leaf", FX_PARAM, FX_INT, NULL,
   "Specifies maximum leaf size in tree. \n" },
  {"lx", FX_PARAM, FX_DOUBLE, NULL,
   "Specifies size of periodic box in x-dimension. \n"},
  {"ly", FX_PARAM, FX_DOUBLE, NULL,
   "Specifies size of periodic box in y-dimension. \n"},
  {"lz", FX_PARAM, FX_DOUBLE, NULL,
   "Specifies size of periodic box in z-dimension. \n"},
  {"bc", FX_PARAM, FX_INT, NULL,
   "Specifies type of boundary conditions. 0 is free boundary, "
   "1 is periodic boundary. \n"},
  {"force_bound", FX_PARAM, FX_DOUBLE, NULL, 
   "Determines pruning criterion for tree implementation \n"},
  FX_ENTRY_DOC_DONE  
};

const fx_module_doc param_doc = {
  param_entries, NULL, 
  "Parameters of Simulated System \n"
};


class DualPhysicsSystem{

  static const int FREE = 0;
  static const int PERIODIC = 1;
  static const int FIXED = 2;
  static const int CUTOFF = 0;
  static const int POTENTIAL = 1;
  static const int FORCE = 2;

private:
  // Input data set
  Matrix atoms_;
  Matrix forces_; 
  Matrix diffusion_;
  // Trees store current state of system
  ParticleTree *system_, *query_;
  ArrayList<int> old_from_new_map_, new_from_old_map_;  

  Vector dimensions_, signs_, powers_;
  
  double time_step_, virial_, temperature_, cutoff_;
  int n_atoms_, boundary_, prune_;
  double n_forces_;
  int total_triples_, range_evals_;
  double max_force_;
  // Tree Parameters
  double force_bound_, percent_pruned_, target_percent_;
  int leaf_size_;
  


  /**
   * Force evaluation functions.
   */ 
 
  // Two-body force between two nodes
  void TwoBodyForce_(ParticleTree* left, ParticleTree* right){ 
    Vector force;
    la::SubInit(right->stat().centroid_, left->stat().centroid_, &force);
    if (boundary_ == PERIODIC){
      AdjustVector_(&force);
    }
    double dist, coef = 0, temp;
    dist = sqrt(la::Dot(force, force));
    for (int i = 0;i < forces_.n_rows(); i++){
      temp = -signs_[i]*left->stat().interactions_[i].coef()*
	right->stat().interactions_[i].coef();
      coef = coef + temp*pow(dist, powers_[i]-2)*powers_[i];
    }     
    virial_ = virial_ + coef*dist*dist;  
    la::Scale(coef*time_step_, &force);    
    left->stat().ApplyForce(force);
    la::Scale(-1.0, &force);
    right->stat().ApplyForce(force);
    percent_pruned_ = percent_pruned_ + 1;
 
  }

  // Two body force between two atoms
  void TwoBodyForce_(int left, int right){   
    Vector delta_r, left_vec, right_vec;    
    atoms_.MakeColumnSubvector(left, 0, 3, &left_vec);
    atoms_.MakeColumnSubvector(right, 0, 3, &right_vec);
    la::SubInit(right_vec, left_vec, &delta_r);
    if (boundary_ == PERIODIC){
      AdjustVector_(&delta_r);
    }
    double dist = sqrt(la::Dot(delta_r, delta_r));
    if (prune_ == CUTOFF){
      if (sqrt(dist ) > cutoff_){
	return;
      }
    }
    double coef = 0, temp;
    for (int i  = 0; i < forces_.n_rows(); i++){
      temp = -forces_.get(i, left)*forces_.get(i, right);
      coef = coef + signs_[i]*temp*powers_[i]*pow(dist, powers_[i]-2);
    }     
  
    virial_ = virial_ +  coef*dist*dist;   
    la::Scale(coef, &delta_r);      
    // Apply forces to particles
    left_vec.Destruct();
    right_vec.Destruct();
    atoms_.MakeColumnSubvector(left, 4, 3, &left_vec);
    atoms_.MakeColumnSubvector(right, 4, 3, &right_vec);
    la::AddExpert(time_step_ / atoms_.get(3,left), delta_r, &left_vec);
    la::AddExpert(-time_step_ / atoms_.get(3,right), delta_r, &right_vec);
    percent_pruned_ = percent_pruned_ + 1;   
 
  }

  
  /**
   * Force bounding functions
   */


  double GetForceTerm_(double R, double r, double Rnorm, double rnorm,
			    int nu){
    double result  = Rnorm*(pow(1-r, -nu-1) -1) / pow(R, nu+2);
    result = result + nu*rnorm / ((nu+2)*pow(R*(1-r), nu+2));
    return result;
  }

  double GetForceTermPt_(double R, double r, double Rnorm, double rnorm,
			    int nu){   
    double result  = Rnorm*(pow(1-r, -nu-1) -1 -(nu+1)*r) / pow(R, nu+2);
    result = result + nu*rnorm / ((nu+2)*pow(R, nu+2))*(1/pow(1-r, nu+2)-1); 
    return result;
  }
  

  double GetPotentialTerm_(double R, double r, int nu){
    double result;
    if (unlikely(r >= 1)){
      return BIG_BAD_NUMBER;
    } else {
      result = (1 / pow(1-r, nu)) /(nu*pow(R, nu));
      return result;
    }
  }

  double GetPotentialTermPt_(double R, double r, int nu){
    double result;
    if (unlikely(r >= 1)){
      return BIG_BAD_NUMBER;
    } else {
      result = (1 / pow(1-r, nu) - nu*r) /(nu*pow(R, nu));
      return result;
    }
  }


  void GetForceRangeDual_(ParticleTree* query, ParticleTree* ref,
			  Vector* bounds){       
    double range_q = 0, range_r = 0;
    double  rnorm = 0, Rnorm = 0;
    Vector delta;
    la::SubInit(query->stat().centroid_, ref->stat().centroid_, &delta);
    AdjustVector_(&delta);
    double Rad = sqrt(la::Dot(delta, delta));

    Vector node_r;
    node_r.Init(3);
    for (int i = 0; i < 3; i++){
      node_r[i] = (query->bound().width(i, dimensions_[i]) + 
		   ref->bound().width(i, dimensions_[i]))/ 2;
      rnorm = rnorm + node_r[i];   
      Rnorm = Rnorm + fabs(delta[i]);
    }
    double rad = sqrt(la::Dot(node_r, node_r))/Rad;
    
  
    for (int i = 0; i < forces_.n_rows(); i++){
      int power = abs((int)powers_[i]);  
      double coef = fabs(ref->stat().interactions_[i].coef()*
			 query->stat().interactions_[i].coef()*
			 GetForceTerm_(Rad, rad, Rnorm, rnorm, power));
      range_q = range_q + coef;
      range_r = range_r + coef;
    }    

   
    range_q = range_q / query->count();
    range_r = range_r / ref->count();

    Vector err; 
    err.Init(2);
    err[0] = fabs(range_q);
    err[1] = fabs(range_r);
  
    la::ScaleOverwrite(time_step_, err, bounds);
  }


  void GetPotentialRangeDual_(ParticleTree* query, ParticleTree* ref,
			  Vector* bounds){       
    double range_q = 0, range_r = 0;   
    Vector delta;
    la::SubInit(query->stat().centroid_, ref->stat().centroid_, &delta);
    AdjustVector_(&delta);
    double Rad = sqrt(la::Dot(delta, delta));

    Vector node_r;
    node_r.Init(3);
    for (int i = 0; i < 3; i++){
      node_r[i] = (query->bound().width(i, dimensions_[i]) + 
		   ref->bound().width(i, dimensions_[i]))/ 2;     
    }
    double rad = sqrt(la::Dot(node_r, node_r)) / Rad;
    
    double coef;
    for (int i = 0; i < forces_.n_rows(); i++){
      int power = abs((int)powers_[i]);  
      coef = fabs(ref->stat().interactions_[i].coef()*
			 query->stat().interactions_[i].coef()*
			 GetPotentialTerm_(Rad, rad, power));
      range_q = range_q + coef;
      range_r = range_r + coef;
    }    
  
    range_q = range_q / query->count();
    range_r = range_r / ref->count();

    Vector err; 
    err.Init(2);
    err[0] = fabs(range_q);
    err[1] = fabs(range_r);
  
    la::ScaleOverwrite(time_step_, err, bounds);
  }
 

 /**
   * Routines for calling force evaluations
   */
  
 

  // This will also cover overlap cases near the diagonal,
  // so query and ref may be the same node. We can evalute three
  // body forces between these two nodes by considering each triple.
  void EvaluateLeafForcesDual_(ParticleTree* query, ParticleTree* ref){
    // Two and Three Body
     for(int i = query->begin(); i < query->begin()+query->count(); i++){
      for(int j = ref->begin(); j < ref->count() + ref->begin(); j++){
	TwoBodyForce_(i,j);	
      }           
    }
  }

  void EvaluateLeafForcesSame_(ParticleTree* query){
    for (int i = query->begin(); i < query->begin() + query->count(); i++){
      for (int j = i+1; j < query->begin() + query->count(); j++){
	TwoBodyForce_(i,j);
      }
    }
  }


  // End Force Evaluation Routines.



  void SplitDual_(ParticleTree* query, ParticleTree* ref, ForceError* err_q,
		  ForceError* err_r){
    ForceError err_q2;
    err_q2.Copy(err_q);    
    double d1, d2;
    if (boundary_ == PERIODIC){
      d1 = ref->bound().PeriodicMinDistanceSq(query->left()->bound(), 
					      dimensions_);
      d2 = ref->bound().PeriodicMinDistanceSq(query->right()->bound(), 
					      dimensions_);
    } else {
      d1 = ref->bound().MinDistanceSq(query->left()->bound());
      d2 = ref->bound().MinDistanceSq(query->right()->bound());  
    }
    if (d1 > d2){    
      UpdateMomentumDual_(query->left(), ref, err_q, err_r);
      UpdateMomentumDual_(query->right(), ref, &err_q2, err_r);
    } else {    
      UpdateMomentumDual_(query->right(), ref, &err_q2, err_r);
      UpdateMomentumDual_(query->left(), ref, err_q, err_r);
    }    
    err_q->Merge(err_q2);
  }
  
   
  int GetPrune_(ParticleTree* i, ParticleTree* j,
		ForceError* err_i, ForceError* err_j){
    int result = 0;
    if (prune_ == CUTOFF){
      double a_min;   
      if (boundary_ == PERIODIC){
	a_min = sqrt(i->bound().PeriodicMinDistanceSq(j->bound(),dimensions_));
      } else {
	a_min = sqrt(i->bound().MinDistanceSq(j->bound()));   
      }      
      result = (a_min > cutoff_);     
    } else {
      Vector range;
      range.Init(2);
      int c1, c2;
      c1 = j->count();
      c2 = i->count();     
      if (prune_ == POTENTIAL){
	GetPotentialRangeDual_(i,j,&range);
      } else {
	GetForceRangeDual_(i,j,&range);
     }
      result = err_i->Check(range[0], c1) * err_j->Check(range[1], c2);
      if (result > 0){
	TwoBodyForce_(i, j);
	err_i->AddVisited(range[0], c1);
	err_j->AddVisited(range[1], c2);	
      }      
    }    
    return result;
  }

    /**
   * Momentum updating routines.
   */
  void UpdateMomentumDual_(ParticleTree* query, ParticleTree* ref,
			   ForceError* err_q, ForceError* err_r){  
    if (GetPrune_(query, ref, err_q, err_r) == 0){
      // Or do we recurse down further?
      int a,b;
      a = query->count();
      b = ref->count();  
      if (a >= b & a > leaf_size_){
	SplitDual_(query, ref, err_q, err_r);	
      } else {
	if (b > leaf_size_){
	  SplitDual_(ref, query, err_r, err_q);	
	} else {	  
	  // Base Case
	  EvaluateLeafForcesDual_(query, ref);
	  // Update Error Terms	 
	  err_r->AddVisited(0, ref->count());
	  err_q->AddVisited(0, query->count());
	}	 
      }
    }
  }


  void UpdateMomentumMain_(ParticleTree* query, ForceError* err_q){
    if (!query->is_leaf()){
      ForceError err_q2;
      err_q2.Copy(err_q);
      UpdateMomentumMain_(query->left(), err_q);
      UpdateMomentumMain_(query->right(), &err_q2);     
      UpdateMomentumDual_(query->left(), query->right(), err_q, &err_q2);
      err_q->Merge(err_q2);      
    } else {
      EvaluateLeafForcesSame_(query);
      err_q->AddVisited(0, (query->count()-1.0));
    }
  }
 
 
  void UpdateMomentumNaive_(){
    for (int i = 0; i < n_atoms_; i++){
      for (int j = i+1; j < n_atoms_; j++){
	TwoBodyForce_(i,j);
      }
    }
  }

  // End momentum udpating routines.
  


  /**
   * Position Updating functions
   */

  /**
   * Update centroids and bounding boxes. Note that we may develop
   * intersections between bounding boxes as the simulation progresses.
   * The velocity of parent nodes is passed down the tree to the leaves.
   */
  void UpdatePositionsRecursion_(ParticleTree* node, Vector* vel){    
    if (likely(!node->is_leaf())){
      Vector temp1;
      Vector temp2;
      temp1.Init(3);
      temp2.Init(3);
      node->stat().GetVelocity(&temp1);
      node->stat().GetVelocity(&temp2);     
      la::AddTo(*vel, &temp1);  
      la::AddTo(*vel, &temp2);
      UpdatePositionsRecursion_(node->left(), &temp1);
      UpdatePositionsRecursion_(node->right(), &temp2);     
      node->stat().UpdateCentroid(node->left()->stat(),node->right()->stat());
      node->bound().Reset();
      node->bound() |= node->left()->bound();
      node->bound() |= node->right()->bound();      
    } else {  // Base Case                
    node->bound().Reset();     
      Vector node_pos, node_vel;
      node_pos.Init(3);
      node_pos.SetZero();
      node_vel.Init(3);
      node_vel.SetZero();     
      for(int i = node->begin(); i < node->begin() + node->count(); i++){
	Vector pos, temp;
	atoms_.MakeColumnSubvector(i, 4, 3, &temp);	
	la::AddTo(*vel, &temp);
	la::AddExpert(atoms_.get(3, i), temp, &node_vel);
	atoms_.MakeColumnSubvector(i, 0, 3, &pos);
	la::AddExpert(time_step_, temp, &pos);
	la::AddExpert(atoms_.get(3, i), pos, &node_pos);
	node->bound() |= pos;
      }            
      node->stat().UpdateKinematics(node_pos, node_vel);
    }      
  }
    
  void UpdatePositionsNaive_(){
    for (int i = 0; i < n_atoms_; i++){
      Vector pos, temp;
      atoms_.MakeColumnSubvector(i, 4, 3, &temp);
      atoms_.MakeColumnSubvector(i, 0, 3, &pos);
      la::AddExpert(time_step_, temp, &pos);      
    }
  }

  void UpdateAtomPosition_(int atom){
    Vector pos, vel;
    atoms_.MakeColumnSubvector(atom, 0, 3, &pos);
    atoms_.MakeColumnSubvector(atom, 4, 3, &vel);
    la::AddExpert(time_step_, vel, &pos);
  }

  // End position updating functions

 
  void InitializeTreeStats_(ParticleTree* node){
    if (!node->is_leaf()){
      InitializeTreeStats_(node->left());
      InitializeTreeStats_(node->right());
      node->stat().MergeStats(node->left()->stat(),node->right()->stat());
    } else{     
      node->stat().InitStats(atoms_, forces_, powers_);          
    }
  }

  void AdjustVector_(Vector* vector_in){
    for(int i = 0; i < 3; i++){
      (*vector_in)[i] = (*vector_in)[i] - dimensions_[i]*
	floor((*vector_in)[i] / dimensions_[i] +0.5);
    }
  } 

  void AdjustVector_(Vector& vector_in, Vector& diff){
    for(int i = 0; i < 3; i++){     
      diff[i] = -floor(vector_in[i] / dimensions_[i] +0.5);
      if(diff[i] != 0){
	printf("Is this the problem? \n");
      }
      vector_in[i] = vector_in[i] + dimensions_[i]*diff[i];     
    }
  }


  void RaddistInternal_(RadDist* raddist, ParticleTree* query){
    if (query->is_leaf()) {   
      RaddistInternalLeaf_(raddist, query);
    } else {
      RaddistExternal_(raddist, query->left(), query->right());
      RaddistInternal_(raddist, query->left());
      RaddistInternal_(raddist, query->right());
    }
  }

  void RaddistExternal_(RadDist* raddist, ParticleTree* query, 
			ParticleTree* ref){    
    double r_min;
    if (boundary_ == PERIODIC){
      r_min = sqrt(query->bound().PeriodicMinDistanceSq(ref->bound(),
							dimensions_));
    } else {    
      r_min = sqrt(query->bound().MinDistanceSq(ref->bound()));
    }    
    if (raddist->GetMax() < r_min){
      return;
    } else {
      // Recurse further	
      if (query->left() != NULL){
	if (ref->left() != NULL){
	  RaddistExternal_(raddist, query->left(), ref->left());
	  RaddistExternal_(raddist, query->left(),ref->right());    
	  RaddistExternal_(raddist, query->right(),ref->left());  
	  RaddistExternal_(raddist, query->right(),ref->right());
	} else {
	  RaddistExternal_(raddist, query->left(), ref); 
	  RaddistExternal_(raddist, query->right(), ref);  	  
	}
      } else{
	if (ref->left() != NULL){
	  RaddistExternal_(raddist, query, ref->left()); 
	  RaddistExternal_(raddist, query, ref->right());	 
	} else {      
	  RaddistExternalLeaf_(raddist, query, ref);        
	}
      }
    }  
  }
  
  void RaddistInternalLeaf_(RadDist* raddist, ParticleTree* query){
    for (int i = query->begin(); i < query->begin() + query->count(); i++){
      for (int j = i+1; j < query->begin() + query->count(); j++){
	Vector left_vec, right_vec, delta_r;
	atoms_.MakeColumnSubvector(i, 0, 3, &left_vec);
	atoms_.MakeColumnSubvector(j, 0, 3, &right_vec);
	la::SubInit(right_vec, left_vec, &delta_r);
	if (boundary_ == PERIODIC){
	  AdjustVector_(&delta_r);
	}
	double dist = sqrt(la::Dot(delta_r, delta_r));
	raddist->Add(dist);
      }
    }
  }

  void RaddistExternalLeaf_(RadDist* raddist, ParticleTree* query, 
			    ParticleTree* ref){
    for (int i = query->begin(); i < query->begin() + query->count(); i++){
      for (int j = ref->begin(); j < ref->begin() + ref->count(); j++){
	Vector left_vec, right_vec, delta_r;
	  atoms_.MakeColumnSubvector(i, 0, 3, &left_vec);
	  atoms_.MakeColumnSubvector(j, 0, 3, &right_vec);
	  la::SubInit(right_vec, left_vec, &delta_r);
	  if (boundary_ == PERIODIC){
	    AdjustVector_(&delta_r);
	  }
	  double dist = sqrt(la::Dot(delta_r, delta_r));
	  raddist->Add(dist);
      }
    }
  }
    
    ///////////////////////////// Constructors ////////////////////////////////

    FORBID_ACCIDENTAL_COPIES(DualPhysicsSystem);

public:
     
    DualPhysicsSystem(){
      system_ = NULL;
      query_ = NULL;
    }
    
    
    ~DualPhysicsSystem(){     
      delete system_;
    }
    
  ////////////////////////////// Public Functions ////////////////////////////

  void Init(const Matrix& atoms_in, struct datanode* param){      
    atoms_.Copy(atoms_in);
    n_atoms_ = atoms_.n_cols();
    diffusion_.Init(3, n_atoms_);
    diffusion_.SetZero();
    force_bound_ = fx_param_double(param, "force_bound", 1.0e-3);    
    leaf_size_ = fx_param_int(param, "leaf", 4);
    Vector dims;
    dims.Init(3);
    dims[0] = 0;
    dims[1] = 1;
    dims[2] = 2;
    boundary_ = fx_param_int(param, "bc", PERIODIC);
    dimensions_.Init(3);
    dimensions_[0] = fx_param_double(param, "lx", 60);
    dimensions_[1] = fx_param_double(param, "ly", 60);
    dimensions_[2] = fx_param_double(param, "lz", 60);  
    system_ = tree::MakeKdTreeMidpointSelective<ParticleTree>(atoms_, dims, 
      leaf_size_, &new_from_old_map_, &old_from_new_map_);   
    cutoff_ = fx_param_double(param, "cutoff", -1);     
    n_forces_ = n_atoms_-1;    
    prune_ = fx_param_int(param, "prune", FORCE);
  } //Init


  
 /**
   * Naive implementation computes all pairwise interactions, and can be 
   * used to validate approximations made by tree implementation.
   */
  void InitNaive(const Matrix& atoms_in, struct datanode* param){      
    atoms_.Copy(atoms_in);
    diffusion_.Init(0,0);
    n_atoms_ = atoms_.n_cols();      
    boundary_ = fx_param_int(param, "bc", FREE);
    dimensions_.Init(3);
    dimensions_[0] = fx_param_double(param, "lx", 60);
    dimensions_[1] = fx_param_double(param, "ly", 60);
    dimensions_[2] = fx_param_double(param, "lz", 60);
    system_ = NULL; 
    old_from_new_map_.Init(0);    
    new_from_old_map_.Init(0);   
  } // InitNaive



  /**
   * Used to initialize electrostatics parameters, for both naive and 
   * tree-based instances of physics system.
   */  
  void InitStats(const Matrix& stats_in, const Vector& signs_in, 
		 const Vector& power_in){        
    powers_.Init(stats_in.n_rows());
    signs_.Init(stats_in.n_rows());
    powers_.CopyValues(power_in);
    signs_.CopyValues(signs_in);
    if (system_ != NULL){     
      forces_.Init(stats_in.n_rows(), stats_in.n_cols());
      // Reindex cols of stats matrix.
      for (int i = 0; i < n_atoms_; i++){
	int k = old_from_new_map_[i];
	for (int j = 0; j < forces_.n_rows(); j++){
	  forces_.set(j, k, stats_in.get(j, i));
	}		
      }
      InitializeTreeStats_(system_);
    } else {
      forces_.Copy(stats_in); 
    }   
  }

  void ReinitStats(const Matrix& stats_in){
    forces_.Destruct();
    forces_.Init(stats_in.n_rows(), stats_in.n_cols());
    // Reindex cols of stats matrix.
    for (int i = 0; i < n_atoms_; i++){
      int k = old_from_new_map_[i];
      for (int j = 0; j < forces_.n_rows(); j++){
	forces_.set(j, k, stats_in.get(j, i));
      }		
    }
    InitializeTreeStats_(system_);    
  }

 

  void RadialDistribution(RadDist* raddist){
    RaddistInternal_(raddist, system_);
    raddist->Scale(2.0 / n_atoms_);
  }

  double GetPercent(){
    return percent_pruned_;
  }

  int GetTrips(){
    return total_triples_;
  }


  void UpdatePositions(double time_step_in){    
    time_step_ = time_step_in;
    Vector temp;
    temp.Init(3);
    temp.SetZero();     
    if (system_ != NULL){
      UpdatePositionsRecursion_(system_, &temp);     
    } else {
      UpdatePositionsNaive_();
    }    
  }     


  void RebuildTree(){
    printf("\n********************************************\n");
    printf("* Rebuilding Tree...\n");
    printf("********************************************\n");
    ArrayList<int> temp_old_new, temp_new_old;
    Vector dims;
    dims.Init(3);
    dims[0] = 0;
    dims[1] = 1;
    dims[2] = 2;   
    for (int i = 0; i < n_atoms_; i++){
      Vector pos, temp, diff;    
      atoms_.MakeColumnSubvector(i, 0, 3, &pos);
      temp.Init(3);
      AdjustVector_(pos, temp);    
      diffusion_.MakeColumnVector(i, &diff);
      la::AddTo(temp, &diff);
    }    
    system_ = tree::MakeKdTreeMidpointSelective<ParticleTree>(atoms_, dims, 
                leaf_size_, &temp_new_old, &temp_old_new);
    Matrix old_diff;
    old_diff.Init(3, n_atoms_);
    old_diff.CopyValues(diffusion_);

    for (int i = 0; i < n_atoms_; i++){
      for (int j = 0; j < 3; j++){
	int k = temp_old_new[i];
	diffusion_.set(j, k, old_diff.get(j,i));
      }      
    }
    for (int i = 0; i < n_atoms_; i++){
      old_from_new_map_[i] = temp_old_new[old_from_new_map_[i]]; 
    }      			      
  }

  double ComputePressure(){    
    // printf("Virial: %f \n", virial_);   
    double pressure = (n_atoms_*temperature_ + virial_) /  
      (3.0*dimensions_[0]*dimensions_[1]*dimensions_[2]); 
    return pressure;
  }
    
 
  void UpdateMomentum(double time_step_in){  
    max_force_ = 0;
    virial_ = 0;    
    time_step_ = time_step_in;
    if (system_ != NULL){
      total_triples_ = 0;
      range_evals_ = 0;    
      percent_pruned_ = 0;       
      ForceError error_main;
      error_main.Init(force_bound_, n_forces_);
      UpdateMomentumMain_(system_, &error_main);    
      percent_pruned_ = 1.0-2*percent_pruned_/(n_atoms_*n_atoms_ - n_atoms_);  
    } else {
      UpdateMomentumNaive_();
    }
  } //UpdateMomentum


  // Compute the average kinetic energy of the particles
  double ComputeTemperature(){
    temperature_ = 0;
    double ke;      
    for(int i = 0; i < n_atoms_; i++){
      Vector vel;
      atoms_.MakeColumnSubvector(i, 4, 3, &vel);
      ke = la::Dot(vel, vel) * atoms_.get(3, i);
      temperature_ = temperature_ + ke;
    }         
    temperature_ = temperature_ / n_atoms_;
    return temperature_;    
  } //ComputeTemperature

 

  void ScaleToTemperature(double temp_in){   
    double ratio;
    ratio = temp_in / temperature_;
    ratio = sqrt(ratio);
    for (int i = 0; i < n_atoms_; i++){
      Vector vel;
      atoms_.MakeColumnSubvector(i, 4, 3, &vel);
      la::Scale(ratio, &vel);
    }	 
  }


  /**
   * When called for a tree implementation, this function finds
   * the RMS deviation between tree and naive results.
   */
  void CompareToNaive(DualPhysicsSystem* comp_sys){  
    if (system_ == NULL){
      return;
    }
    double rms_deviation = 0;
    int i;
    for (i = 0; i < n_atoms_; i++){    
      Vector pos_1, pos_2, delta_r;
      int j = old_from_new_map_[i];
      atoms_.MakeColumnSubvector(j, 0, 3, &pos_1);
      comp_sys->atoms_.MakeColumnSubvector(i, 0, 3, &pos_2);
      la::SubInit(pos_1, pos_2, &delta_r);
      if (boundary_ == PERIODIC){
	AdjustVector_(&delta_r);
      }
      rms_deviation = rms_deviation + la::Dot(delta_r, delta_r);; 
    }    
    rms_deviation = sqrt(rms_deviation / n_atoms_);  
    printf("rms_deviation: %f \n", rms_deviation);   
  } // CompareToNaive


  double ComputeDiffusion(const Matrix& old_positions){
    double diff = 0.0;
    for (int i = 0; i< n_atoms_; i++){
      Vector oldp, newp, del;
      int j = i;
      if (system_ != NULL){
	j = old_from_new_map_[j];
      }
      atoms_.MakeColumnSubvector(j, 0, 3, &newp);
      del.Init(3);
      del.SetZero();
      for (int k = 0; k < 3; k++){
	del[k] = newp[k] - dimensions_[k]*diffusion_.get(k,j);
      }
      old_positions.MakeColumnSubvector(i, 0, 3, &oldp);
      la::AddExpert(-1.0, oldp, &del);     
      diff = diff + la::Dot(del, del);
    }
    return diff / n_atoms_;
  }

 
  // Write all atom positions to the specified file
  void WritePositions(FILE* fp){
    int i;
    for (i = 0; i < n_atoms_; i++){
      Vector temp;	
      int j = i;
      if (system_ != NULL){
	j = old_from_new_map_[j];
      }
      atoms_.MakeColumnSubvector(j, 0, 3, &temp);
      if (boundary_ == PERIODIC){
	AdjustVector_(&temp);
      }
      fprintf(fp, " %16.8f, %16.8f, %16.8f \n", temp[0], temp[1], temp[2]);   
    }   
  } // WritePositions

 void RecordPositions(Matrix& out_positions){
    for (int i = 0; i < n_atoms_; i++){
      Vector temp, temp2;
      int j = i;
      if (system_ != NULL){
	j = old_from_new_map_[j];
      }
      atoms_.MakeColumnSubvector(j, 0, 3, &temp);
      diffusion_.MakeColumnVector(j, &temp2);
      for (int k = 0; k < 3; k++){
	out_positions.set(k , i, temp[k]-temp2[k]*dimensions_[k]);
      }
    }
  }

  void WriteMomentum(FILE* fp){   
    for (int i = 0; i < n_atoms_; i++){
      Vector temp;
      int j = i;
      if (system_ != NULL){
	j = old_from_new_map_[j];
      }
      atoms_.MakeColumnSubvector(j, 4, 3, &temp);
      fprintf(fp, " %16.8f, %16.8f, %16.8f \n", temp[0], temp[1], temp[2]);
    }
  } 

  void WriteData(FILE* fp){
    for (int i = 0; i < n_atoms_; i++){
      Vector temp;
      int j = i;
      if (system_ != NULL){
	j = old_from_new_map_[i];	
      }
      atoms_.MakeColumnSubvector(j, 0, 3, &temp);
      fprintf(fp, " %16.8f, %16.8f, %16.8f, ", temp[0], temp[1], temp[2]);
      double mass = atoms_.get(3, j);
      fprintf(fp, "%16.8f,", mass);
      temp.Destruct();
      atoms_.MakeColumnSubvector(j, 4, 3, &temp);
      fprintf(fp, " %16.8f, %16.8f, %16.8f \n ", temp[0], temp[1], temp[2]);
    }
  }

}; // class PhysicsSystem

#endif

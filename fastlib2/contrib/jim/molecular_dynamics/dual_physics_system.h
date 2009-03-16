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

private:
  // Input data set
  Matrix atoms_;
  Matrix forces_; 
  Matrix diffusion_;
  // Trees store current state of system
  ParticleTree *system_, *query_;
  ArrayList<int> old_from_new_map_, new_from_old_map_;  

  Vector dimensions_, signs_, powers_;
  
  
  double time_step_, boundary_, virial_, temperature_, cutoff_;  
  int n_atoms_;  
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
    if (cutoff_ > 0){
      if (dist  > cutoff_){
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


  double GetForceRangeTerm_(double R, double r, double Rnorm, double rnorm,
			    int nu){
    double  eta = r*r*rnorm;
    double result  = Rnorm*((nu-1+r)/pow(1-r, nu+1) - 0.5*(r*r*nu*(nu-1)*(nu-1)
      + 2*nu*nu*r+2*(nu-1))) / pow(R, nu+2);
    result = result + eta*(1 / pow(1-r, nu) -nu*r-1)/
      ((nu+1)*(nu+2)*pow(R, nu+2));
    return result;
  }




  void GetForceRangeDual_(ParticleTree* query, ParticleTree* ref,
			  Vector* bounds){       
    double range_q = 0, range_r = 0;
    double Rr, Rq, rr, rq, rnormr = 0, rnormq = 0;
    Rr = sqrt(ref->bound().PeriodicMinDistanceSq(query->stat().centroid_, 
						 dimensions_));
    Rq = sqrt(query->bound().PeriodicMinDistanceSq(ref->stat().centroid_,
						   dimensions_));
    Vector br, bq;
    br.Init(3);
    bq.Init(3);
    for (int i = 0; i < 3; i++){
      br[i] = query->bound().width(i, dimensions_[i]) /2;
      bq[i] = ref->bound().width(i, dimensions_[i]) /2;
      rnormr = rnormr + br[i];
      rnormq = rnormq + bq[i];
    }
    rq = sqrt(la::Dot(bq, bq)) / Rq;
    rr = sqrt(la::Dot(br, br)) / Rr;
    double Rnormr = query->bound().PeriodicMaxDistance1Norm(
      ref->stat().centroid_, dimensions_);
    double Rnormq = ref->bound().PeriodicMaxDistance1Norm(
      query->stat().centroid_, dimensions_);
   
    for (int i = 0; i < forces_.n_rows(); i++){
      int power = abs((int)powers_[i]);      
      range_q = range_q + fabs(ref->stat().interactions_[i].coef()*
	signs_[i]*GetForceRangeTerm_(Rq, rq, Rnormq, rnormq, power));
      range_r = range_r + fabs(query->stat().interactions_[i].coef()*
	signs_[i]*GetForceRangeTerm_(Rr, rr, Rnormr, rnormr, power));
    }    
    Vector err;
    err.Init(2);
    err[0] = range_q;
    err[1] = range_r;
    
    la::ScaleOverwrite(time_step_, err, bounds);
  }

    
  /**
   * Force bounding functions
   */
  /*
  void GetForceRangeDual_(ParticleTree* query, ParticleTree* ref,
			  Vector* bounds){       
    double r_min, r_max;
    if (boundary_ == PERIODIC){
      r_min = sqrt(query->bound().PeriodicMinDistanceSq(ref->bound(),
						       dimensions_));
      r_max = sqrt(query->bound().PeriodicMaxDistanceSq(ref->bound(),
						       dimensions_));
    } else {
      r_min = sqrt(query->bound().MinDistanceSq(ref->bound()));     
      r_max = sqrt(query->bound().MaxDistanceSq(ref->bound()));
    }        
    Vector max, min, fmax_qr, fmin_qr, fmax_rq, fmin_rq;
    max.Init(3);
    min.Init(3);
    fmax_qr.Init(3);
    fmin_qr.Init(3);
    fmax_qr.SetZero();
    fmin_qr.SetZero();
    fmax_rq.Init(3);
    fmin_rq.Init(3);
    fmax_rq.SetZero();
    fmin_rq.SetZero();
    for(int i = 0; i < 3; i++){
      max[i] = query->bound().MaxDelta(ref->bound(), dimensions_[i],i);
      min[i] = query->bound().MinDelta(ref->bound(), dimensions_[i],i);
    }
    for (int i = 0; i < forces_.n_rows(); i++){
      int power = (int)powers_[i];
      double temp = query->stat().interactions_[i].coef()*
	ref->stat().interactions_[i].coef()*power*signs_[i];      
      for (int j = 0; j < 3; j++){
	if (max[j]*temp > 0){
	  fmax_qr[j] = fmax_qr[j] + max[j]*temp*pow(r_min, power-2);
	  fmin_rq[j] = fmin_rq[j] - max[j]*temp*pow(r_min, power-2);
	} else {
	  fmax_qr[j] = fmax_qr[j] + max[j]*temp*pow(r_max, power-2);
	  fmin_rq[j] = fmin_rq[j] - max[j]*temp*pow(r_max, power-2);
	}
	if( min[j]*temp > 0){
	  fmin_qr[j] = fmin_qr[j] + min[j]*temp*pow(r_max, power-2);
	  fmax_rq[j] = fmax_rq[j] - min[j]*temp*pow(r_max, power-2);
	} else {
	  fmin_qr[j] = fmin_qr[j] + min[j]*temp*pow(r_min, power-2);
	  fmax_rq[j] = fmax_rq[j] - min[j]*temp*pow(r_min, power-2);	  
	}
      }
    }    
    // Get range from omitting three body interactions 
    
    Vector delta_rq, delta_qr, err;
    la::SubInit(fmax_qr, fmin_qr, &delta_qr);   
    la::SubInit(fmax_rq, fmin_rq, &delta_rq);
    err.Init(2);
    err.SetZero();
    for (int i = 0; i < 3; i++){
      err[0] = err[0] + fabs(delta_qr[i]);
      err[1] = err[1] + fabs(delta_rq[i]);
    }
    err[1] = err[1] / ref->stat().mass_;
    err[0] = err[0] / query->stat().mass_;
    la::ScaleOverwrite(time_step_, err, bounds);   
  }
  */


 int GetForceRangeDualCutoff_(ParticleTree* query, ParticleTree* ref){   
   int result;
   double r_min;
   if (boundary_ == PERIODIC){
     r_min = sqrt(query->bound().PeriodicMinDistanceSq(ref->bound(),
						       dimensions_));      
   } else {
     r_min = sqrt(query->bound().MinDistanceSq(ref->bound()));       
   }        
   result = (r_min > cutoff_);      
   return result;
 }
 


 /* 
     Range of multipole expansion of 1/R^nu, truncated after
     monopole term, from Duan and Krasny.
  */
  double RangeDK_(int nu, double r){
    return (pow(1.0 / (1-r), nu) - 1.0) /nu;
  }
 

  /**
   * Routines for calling force evaluations
   */
  void EvaluateNodeForcesDual_(ParticleTree* query, ParticleTree* ref){   
    TwoBodyForce_(query, ref);
  }


 
  // This will also cover overlap cases near the diagonal,
  // so query and ref may be the same node. 
  void EvaluateLeafForcesDual_(ParticleTree* query, ParticleTree* ref){  
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


  /**
   * Momentum updating routines.
   */
  void UpdateMomentumDual_(ParticleTree* query, ParticleTree* ref){
    if (ref->begin() < query->begin()){
      return;
    }
    Vector force_range, error_bound;   
    error_bound.Init(2);
    force_range.Init(2);
    GetForceRangeDual_(query, ref, &force_range);
    error_bound[0] = force_bound_*ref->count()/ n_atoms_;
    error_bound[1] = force_bound_*query->count()/ n_atoms_;    
    // Can we evaluate force here?
    if (force_range[0] < error_bound[0] & force_range[1] < error_bound[1]){
      GetForceRangeDual_(query, ref, &force_range);
      EvaluateNodeForcesDual_(query, ref);     
    } else {
      // Or do we recurse down further?
      int a,b;
      a = query->count();
      b = ref->count();  
      if (a >= b & a > leaf_size_){
	UpdateMomentumDual_(query->left(), ref);
	UpdateMomentumDual_(query->right(), ref);
      } else {
	if (b > leaf_size_){
	  UpdateMomentumDual_(query, ref->left());
	  UpdateMomentumDual_(query, ref->right());	  
	} else {
	    // Base Case
	  EvaluateLeafForcesDual_(query, ref);
	}	 
      }
    }
  }


  void UpdateMomentumMain_(ParticleTree* query){
    if (!query->is_leaf()){
      UpdateMomentumMain_(query->left());
      UpdateMomentumMain_(query->right());
      if (cutoff_ >0){
	UpdateMomentumDualCutoff_(query->left(), query->right());
      } else {
	UpdateMomentumDual_(query->left(), query->right());
      }
    } else {
      EvaluateLeafForcesSame_(query);
    }
  }
 

  void UpdateMomentumDualCutoff_(ParticleTree* query, ParticleTree* ref){ 
    int prune;   
    prune = GetForceRangeDualCutoff_(query, ref);
    if (!prune){
      // Or do we recurse down further?
      int a,b;
      a = query->count();
      b = ref->count();  
      if (a >= b & a > leaf_size_){
	UpdateMomentumDualCutoff_(query->left(), ref);
	UpdateMomentumDualCutoff_(query->right(), ref);
      } else {
	if (b > leaf_size_){
	  UpdateMomentumDualCutoff_(query, ref->left());
	  UpdateMomentumDualCutoff_(query, ref->right());	  
	} else {
	  // Base Case	  
	  EvaluateLeafForcesDual_(query, ref);	   	  
	}	 
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
      	if (boundary_ == FREE){
	  node->bound() |= node->left()->bound();
	  node->bound() |= node->right()->bound();  	 
	} else {
	  node->bound().Add(node->left()->bound(), dimensions_);
	  node->bound().Add(node->right()->bound(), dimensions_);
	}
        
    } else {  // Base Case                
      node->bound().Reset();     
      la::AddTo(node->stat().velocity_, vel); 
      for(int i = node->begin(); i < node->begin() + node->count(); i++){
	Vector pos, temp;
	atoms_.MakeColumnSubvector(i, 4, 3, &temp);
	la::AddTo(*vel, &temp);
	atoms_.MakeColumnSubvector(i, 0, 3, &pos);
	la::AddExpert(time_step_, temp, &pos);
	if (boundary_ == FREE){
	  node->bound() |= pos;
	} else {
	  node->bound().Add(pos, dimensions_);
	}
      }            
      node->stat().InitKinematics(node->begin(), node->count(), atoms_);
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
      if (system_ != NULL){
	//	delete system_;
      }     
    }
    
  ////////////////////////////// Public Functions ////////////////////////////

  void Init(const Matrix& atoms_in, struct datanode* param){      
    atoms_.Copy(atoms_in);
    diffusion_.Init(3, atoms_.n_cols());
    diffusion_.SetZero();
    n_atoms_ = atoms_.n_cols();
    force_bound_ = fx_param_double(param, "force_bound", 0.001);
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
  } //Init



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

 

  void UpdatePositions(double time_step_in){    
    time_step_ = time_step_in;
    Vector temp;
    temp.Init(3);
    temp.SetZero();     
    UpdatePositionsRecursion_(system_, &temp);     
  }     


  void RebuildTree(){
    //    printf("\n********************************************\n");
    //   printf("* Rebuilding Tree...\n");
    //   printf("********************************************\n");
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
      if (boundary_ == PERIODIC){
	AdjustVector_(pos, temp);    
	diffusion_.MakeColumnVector(i, &diff);
	la::AddTo(temp, &diff);
      }
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
    percent_pruned_ = 0;   
    UpdateMomentumMain_(system_);    
    percent_pruned_ = 1.0-2*percent_pruned_ / (n_atoms_*n_atoms_ - n_atoms_);  
  } //UpdateMomentum


  // Compute the average kinetic energy of the particles
  double ComputeTemperature(){
    temperature_ = 0;
    double ke = 0;      
    for(int i = 0; i < n_atoms_; i++){
      Vector vel;
      atoms_.MakeColumnSubvector(i, 4, 3, &vel);     
      ke = la::Dot(vel,vel)*atoms_.get(3,i);
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
  
  void RecordPositions(Matrix& out_positions){
    for (int i = 0; i < n_atoms_; i++){
      Vector temp;
      int j = i;
      if (system_ != NULL){
	j = old_from_new_map_[j];
      }
      atoms_.MakeColumnSubvector(j, 0, 3, &temp);
      for (int k = 0; k < 3; k++){
	out_positions.set(k , i, temp[k]);
      }
    }
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

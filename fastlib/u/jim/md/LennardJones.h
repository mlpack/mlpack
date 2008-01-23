/**
 * @file LennardJones.h
 *
 * Molecular Dynamics via Lennard-Jones potential, where pairwise forces
 * are given from the gradient of the potential function
 * 4 epsilon*((sigma / r)^12 - (sigma / r)^6) 
 *
 * Accelerations are calculated via single-tree search. 
 * Equations of motion are integrated by a leapfrogging method.
 *
 * @see LennardJones_main.cc
 */


#include "fastlib/fastlib.h"
#include "fastlib/fastlib_int.h"
#include "AtomTree.h"

class TestLennardJones;

class LennardJones{

  // Friend Test Class
  friend class TestLennardJones;

private:
  // Input data set
  Matrix atoms_, velocities_;
  // Trees store current state of system
  AtomTree *system_, *query_;
  ArrayList<int> old_from_new_map_, foo_; 

  // Simulation parameters
  // Values are in eV, Angstroms, and amu, which
  // places time (roughly) in units of nanoseconds.
  // Default values correspond to Argon, taken from Ashcroft & Mermin
  int n_atoms_;
  double epsilon_, sigma_, cutoff_sq_, mass_, time_step;
  

  /**
   * Compute the change in the velocity of center_1, according to Lennard-Jones
   * potential between centers 1 & 2. This force will be scaled when it is 
   * returned to UpdateVelocityRecursion, if center_2 corresponds to more 
   * than one atom. Note also that this computes the acceleration integrated 
   * over the time step,equal to the change in velocity, rather than the 
   * instantaneous acceleration.
   */
  void Acceleration_(Vector &center_1, Vector &center_2, Vector &delta_r_){
    int i;
    for (i = 0; i < 3; i++){
      delta_r_[i] = center_1[i] - center_2[i];
    }
    double dist_sq_ = la::DistanceSqEuclidean(center_1, center_2);
    double r_scaled_ = sigma_*sigma_ / dist_sq_;
    r_scaled_ = r_scaled_*r_scaled_*r_scaled_;
    double force_mag_ = time_step*24*epsilon_*r_scaled_*(2*r_scaled_ - 1) / 
      (dist_sq_*mass_);
    la::Scale(force_mag_, &delta_r_);         
  }

  /**
   * Update centroids and bounding boxes. Note that we may develop
   * intersections between bounding boxes as the simulation progresses.
   */
  void UpdatePositionsRecursion_(AtomTree *current_node){       
    if (likely(current_node->count() > 1)){
      UpdatePositionsRecursion_(current_node->left());
      UpdatePositionsRecursion_(current_node->right());
      current_node->bound().Reset();
      current_node->bound() |= current_node->left()->bound();
      current_node->bound() |= current_node->right()->bound();
      current_node->stat().UpdateCentroid(current_node->left()->stat(), 
					current_node->right()->stat());     
    } else {  // Base Case
      current_node->stat().UpdateCentroid(time_step);
      current_node->bound().Reset();
      current_node->bound() |= current_node->stat().centroid;
    }      
  }

  /**
   * Compute the effect of the reference node on the velocity of
   * the query node. The effect of distant atoms is approximated
   * from the centroid of the reference node atoms.
   */
  void UpdateVelocityRecursion_(AtomTree* vel_query_, AtomTree* vel_ref_){   
    if (unlikely(vel_ref_->count() == 1)){
      UpdateVelocityBase_(vel_query_, vel_ref_);
    } else {
      double dist_sq;
      dist_sq = vel_ref_->bound().MinDistanceSq(vel_query_->bound());
      if (dist_sq > cutoff_sq_){
	Vector delta_v_;
	delta_v_.Init(3);       
	Acceleration_(vel_query_->stat().centroid, 
		      vel_ref_->stat().centroid, delta_v_);
	la::Scale(vel_ref_->stat().mass, &delta_v_);	
	la::AddTo(delta_v_, &vel_query_->stat().velocity);
      } else {       
	UpdateVelocityRecursion_(vel_query_, vel_ref_->right());
	UpdateVelocityRecursion_(vel_query_, vel_ref_->left());
      }
    }
  } //UpdateVelocityRecursion


  /**
   * Base case calculates pairwise interactions between nearby atoms.
   */
  void UpdateVelocityBase_(AtomTree* vel_query_, AtomTree* vel_ref_){
    if (likely(vel_query_->begin() != vel_ref_->begin())){
      Vector delta_v_;
      delta_v_.Init(3);       
      Acceleration_(vel_query_->stat().centroid, 
		    vel_ref_->stat().centroid, delta_v_);
      la::AddTo(delta_v_, &vel_query_->stat().velocity);
    }
  } // UpdateVelocityBase
  
    ///////////////////////////// Constructors ////////////////////////////////

    FORBID_ACCIDENTAL_COPIES(LennardJones);

public:
     
    LennardJones(){
      system_ = NULL;
      query_ = NULL;
    }
    
    
    ~LennardJones(){
      if (system_ != NULL){
	delete system_;
      }      
    }
    
  ////////////////////////////// Public Functions ////////////////////////////

  void Init(const Matrix& atoms_in, struct datanode* param){
    epsilon_ = fx_param_double(param, "eps", 0.0104);   
    sigma_ = fx_param_double(param, "sig", 2.74);
    mass_ = fx_param_double(param, "mass", 40.0);
    cutoff_sq_ = fx_param_double(param, "r_max", 6*sigma_);
    cutoff_sq_ = cutoff_sq_*cutoff_sq_;
    velocities_.Init(1,1);
    atoms_.Copy(atoms_in);
    n_atoms_ = atoms_.n_cols();
    system_ = tree::MakeKdTreeMidpoint<AtomTree>(atoms_, 1, &foo_,
						 &old_from_new_map_);    
  } //Init


  /**
   * Naive implementation computes all pairwise interactions, and can be 
   * used to validate approximations made by tree implementation.
   */
  void InitNaive(const Matrix& atoms_in, struct datanode* param){
    epsilon_ = fx_param_double(param, "eps", 0.0104);   
    sigma_ = fx_param_double(param, "sig", 2.74);
    mass_ = fx_param_double(param, "mass", 40.0);

    atoms_.Alias(atoms_in);
    n_atoms_ = atoms_.n_cols();
    system_ = NULL;
    query_ = NULL;
 
    velocities_.Init(3, n_atoms_);
    velocities_.SetZero();
    old_from_new_map_.Init(0);    
    foo_.Init(0);
  } // InitNaive


  void UpdatePositions(double time_step_in){
    time_step = time_step_in;
    UpdatePositionsRecursion_(system_); 
  }

  /**
   * Naive velocity update computes each pairwise interaction.
   */
  void UpdateVelocities(double time_step_in){
    time_step = time_step_in;
    int i;
    for (i = 0; i < n_atoms_; i++){
      query_ = system_->FindByBeginCount(i,1);      
      UpdateVelocityRecursion_(query_, system_);
    }
  } //UpdateVelocities

  void UpdatePositionsNaive(double time_step_in){
    int i, j;     
    for (i = 0; i < n_atoms_; i++){
      for (j = 0; j < 3; j++){
	atoms_.set(j,i, atoms_.get(j,i) + time_step_in*velocities_.get(j,i));
      }
    }

  } //UpdatePositionsNaive


  void UpdateVelocitiesNaive(double time_step_in){
    time_step = time_step_in;
    int i, j, k;
    Vector delta_v_;
    delta_v_.Init(3);
    for (i = 0; i < n_atoms_; i++){
      for (j = i+1; j < n_atoms_; j++){
	Vector position_i, position_j;
	atoms_.MakeColumnSubvector(i, 0,3, &position_i);
	atoms_.MakeColumnSubvector(j, 0,3, &position_j);
	Acceleration_(position_i, position_j, delta_v_);
	for (k = 0; k < 3; k++){
	  velocities_.set(k, i, velocities_.get(k,i) + delta_v_[k]);
	  velocities_.set(k, j, velocities_.get(k,j) - delta_v_[k]);  
	}
      }
    }

  } //UpdateVelocitiesNaive


  /**
   * When called for a tree implementation, this function finds
   * the RMS deviation between tree and naive results.
   */
  void CompareToNaive(Matrix naive_result){  
    if (system_ == NULL){
      return;
    }
    Vector naive_position;
    double rms_deviation = 0;
    int i;
    for (i = 0; i < n_atoms_; i++){
      query_ = system_->FindByBeginCount(old_from_new_map_[i],1);
      naive_result.MakeColumnVector(i, &naive_position);
      rms_deviation = rms_deviation + 
	la::DistanceSqEuclidean(query_->stat().centroid, naive_position);
      naive_position.Destruct();
    }    
    rms_deviation = sqrt(rms_deviation / n_atoms_);
    printf("RMS Deviation: %f \n", rms_deviation);
    naive_position.Init(0);
  } // CompareToNaive


  // Write all atom positions to the specified file
  void WritePositions(FILE* fp){
    int i;
    // Tree case
    if (system_ != NULL){      
      for (i = 0; i < n_atoms_; i++){
	query_ = system_->FindByBeginCount(old_from_new_map_[i],1);
	fprintf(fp, " %16.8f, %16.8f, %16.8f \n", 
		query_->stat().centroid[0], 
		query_->stat().centroid[1], 
		query_->stat().centroid[2]);
      }      
    } else {  // Naive case
      for (i = 0; i < n_atoms_; i++){
	fprintf(fp, " %16.8f, %16.8f, %16.8f \n", 
		atoms_.get(0, i), atoms_.get(1, i), atoms_.get(2, i));
      }
    }

  } // WritePositions

}; // class LennardJones




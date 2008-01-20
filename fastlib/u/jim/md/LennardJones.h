/**
 * @file LennardJones.h
 *
 * Molecular Dynamics via Lennard-Jones potential
 *
 * Accelerations are calculated via single-tree search. 
 * Equations of motion are integrated by a leapfrogging method.
 *
 * @see LennardJones_main.cc
 */


#include "fastlib/fastlib.h"
#include "fastlib/fastlib_int.h"
#include "AtomTree.h"
#include "col/queue.h"
#include "tree/kdtree.h"
#include "tree/spacetree.h"
#include "tree/bounds.h"
#include "la/la.h"
#include "math/math.h"
#include "data/dataset.h"


class LennardJones{

private:
  // Input data set
  Matrix atoms_, velocities_;
  // Trees store current state of system
  AtomTree *system_, *query_;

  // Simulation parameters
  int n_atoms_;
  double eps, sig, cutoff, mass, time_step;
  

  /**
   * Compute the change in the velocity of center_1, according to Lennard-Jones
   * potential between centers 1 & 2. This force will be scaled when it is returned
   * to UpdateVelocityRecursion, if center_2 corresponds to more than one atom.
   * Note also that this computes the acceleration integrated over the time step,
   * equal to the change in velocity, rather than the instantaneous acceleration.
   */
  void Acceleration_(Vector &center_1, Vector &center_2, Vector &delta_v_){
    int i;
    for (i = 0; i < 3; i++){
      delta_v_[i] = center_1[i] - center_2[i];
    }
    double dist_sq_ = la::DistanceSqEuclidean(center_1, center_2);
    double r_scaled_ = sig*sig / dist_sq_;
    r_scaled_ = r_scaled_*r_scaled_*r_scaled_;
    double force_mag_ = time_step*24*eps*r_scaled_*(2*r_scaled_ - 1) / (dist_sq_*mass);
    la::Scale(force_mag_, &delta_v_);         
  }

  /**
   * Update centroids and bounding boxes. Note that we may develop
   * intersections between bounding boxes as the simulation progresses.
   */
  void UpdatePositionsRecursion_(AtomTree *current_node){
    int i;     
    if (likely(current_node->count() > 1)){
      UpdatePositionsRecursion_(current_node->left());
      UpdatePositionsRecursion_(current_node->right());
      current_node->bound().Reset();
      current_node->bound() |= current_node->left()->bound();
      current_node->bound() |= current_node->right()->bound();
      for (i = 0; i < 3; i++){
	current_node->stat().centroid[i] = 
	  current_node->left()->stat().mass*current_node->left()->stat().centroid[i] + 
	  current_node->right()->stat().mass*current_node->right()->stat().centroid[i];
	current_node->stat().centroid[i] = 
	  current_node->stat().centroid[i] / current_node->stat().mass;
      } 
    } else {  // Base Case
      for (i = 0; i < 3; i++){
	current_node->stat().centroid[i] = current_node->stat().centroid[i] + 
	  time_step * current_node->stat().velocity[i];	
      }
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
      if (dist_sq > cutoff){
	Vector delta_v_;
	delta_v_.Init(3);       
	Acceleration_(vel_query_->stat().centroid, vel_ref_->stat().centroid, delta_v_);
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
      Acceleration_(vel_query_->stat().centroid, vel_ref_->stat().centroid, delta_v_);
      la::AddTo(delta_v_, &vel_query_->stat().velocity);
    }
  } // UpdateVelocityBase


  ////////////////////////////// Constructors ///////////////////////////////////////

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
    
  ////////////////////////////// Public Functions ///////////////////////////////////

  void Init(const Matrix& atoms_in, double eps_in, double sig_in, double mass_in, double cutoff_in){
    velocities_.Init(1,1);
    atoms_.Copy(atoms_in);
    n_atoms_ = atoms_.n_cols();
    system_ = tree::MakeKdTreeMidpoint<AtomTree>(atoms_, 1);
    eps = eps_in;
    sig = sig_in;
    mass = mass_in;
    cutoff = cutoff_in;

  } //Init


  /**
   * Naive implementation computes all pairwise interactions, and can be used to
   * validate approximations made by tree implementation.
   */
  void InitNaive(const Matrix& atoms_in, double eps_in, double sig_in, double mass_in){

    atoms_.Copy(atoms_in);
    n_atoms_ = atoms_.n_cols();
    system_ = NULL;
    query_ = NULL;
    eps = eps_in;
    sig = sig_in;
    mass = mass_in;
    velocities_.Init(3, n_atoms_);
    velocities_.SetZero();
  } // InitNaive


  void UpdatePositions(double time_step_in){
    time_step = time_step_in;
    UpdatePositionsRecursion_(system_); 

  }

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


  void WritePositions(FILE* fp){
    int i;
    // Tree case
    if (system_ != NULL){      
      for (i = 0; i < n_atoms_; i++){
	query_ = system_->FindByBeginCount(i,1);
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




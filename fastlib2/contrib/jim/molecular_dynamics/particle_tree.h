/**
 * @file particle_tree.h
 *
 * @author Jim Waters (jwaters6@gatech.edu)
 *
 * General tree structure for physics problems. Each PhysStat instance
 * represents a particular kind of interaction affecting this particle.
 * This stat houses the kinematic quantities (center of mass, momentum, and 
 * mass) used by all particles for computing dynamics.
 * 
 */

#ifndef PARTICLE_TREE_H
#define PARTICLE_TREE_H

#include "fastlib/fastlib.h"
#include "two_body_stat.h"
#include "three_body_stat.h"

struct ParticleStat { 
  friend class TestParticleStat;

  Vector centroid_;
  Vector velocity_; 
  double mass_, radius_, error_;
  ArrayList<TwoBodyStat> interactions_;
  Vector axilrod_;
  Vector delta_, temp_delta_;
  int start_, count_;
  

  /**
   * Default Initialization
   */
  void Init(){   
  }

  /**
   * Init function for leaf node.
   */
  void Init(const Matrix& dataset, int start, int count){    
    start_ = start;
    count_ = count;  
    InitKinematics(start, count, dataset);  
    interactions_.Init(0);
    axilrod_.Init(2);   
  }  

  /**
   * Init function to build node from two children, tracking mass and
   * centroid of each node.
   */
  void Init(const Matrix& dataset, int start, int count, 
	    ParticleStat &left_stat, ParticleStat &right_stat){  
    start_ = start;
    count_ = count;		 
    mass_ = left_stat.mass_ + right_stat.mass_;
    velocity_.Init(3);
    velocity_.SetZero();   
    la::ScaleInit(left_stat.mass_ / mass_, left_stat.centroid_, &centroid_);
    la::AddExpert(right_stat.mass_ /mass_, right_stat.centroid_, &centroid_);  
    Vector temp;
    la::ScaleInit(-1.0, left_stat.centroid_, &temp);
    la::AddTo(right_stat.centroid_, &temp);
    radius_ = sqrt(la::Dot(temp, temp));
    radius_ = radius_ + left_stat.radius_ + right_stat.radius_;    
    axilrod_.Init(2);
  }


  void InitStats(const Matrix& dataset, const Matrix& forces, 
		const Vector& powers){
    for(int i = 0; i < forces.n_rows(); i++){
      interactions_.PushBack(1);
      double coef = 0, abs_coef = 0;    
      for (int j = start_; j < start_ + count_; j++){
	abs_coef = abs_coef + fabs(forces.get(i, j));
	coef = coef + forces.get(i, j);
      }     
      interactions_[interactions_.size() - 1].Init(coef, (int)powers[i]);     
    }
  }


  void InitStat(const Vector& center, double coef, int power){
    interactions_.PushBack(1);
    interactions_[interactions_.size() - 1].Init(coef, power);  
  }

  void InitAT(Matrix* matrix_in){
    axilrod_.SetZero();
    for (int i = start_; i < start_ + count_; i++){
      axilrod_[0] = axilrod_[0] + matrix_in->get(0,i);
      axilrod_[1] = axilrod_[1] + matrix_in->get(1,i);
    }
  }

  void MergeAT(ParticleStat &left_stat, ParticleStat& right_stat){
    axilrod_[0] = left_stat.axilrod_[0] + right_stat.axilrod_[0];
    axilrod_[1] = left_stat.axilrod_[1] + right_stat.axilrod_[1];    
  }

  void MergeStats(const ParticleStat& left_stat, 
		  const ParticleStat& right_stat){    
    interactions_.Init(left_stat.interactions_.size());
     for (int i = 0; i < interactions_.size(); i++){           
      interactions_[i].Init(left_stat.interactions_[i], 
			    right_stat.interactions_[i]);
    }
  }


  /**
   * Initialization function
   */
  void InitKinematics(int start, int count, const Matrix& dataset){  
    centroid_.Init(3);
    centroid_.SetZero();
    velocity_.Init(3);
    velocity_.SetZero();
    mass_ = 0;
    for (int i = start_; i < start_+count_; i++){    
      mass_ = mass_ + dataset.get(3, i);
      Vector temp;
      dataset.MakeColumnSubvector(i, 0, 3, &temp);      
      la::AddExpert(dataset.get(3, i), temp, &centroid_);    
    }  
    la::Scale(1.0 / mass_, &centroid_);   
    for (int i = start_; i < start_ + count_; i++){
      double new_rad = 0;
      Vector temp1, temp2;
      dataset.MakeColumnSubvector(i, 0, 3, &temp1);
      la::SubInit(centroid_, temp1, &temp2);     
      new_rad = la::Dot(temp2, temp2);
      if (new_rad > radius_){
	radius_ = new_rad;
      }     
    }  
    radius_ = sqrt(radius_);
  }

  void UpdateCentroid(const ParticleStat& left, const ParticleStat& right){
    centroid_.Destruct();
    la::ScaleInit(left.mass_, left.centroid_, &centroid_);    
    la::AddExpert(right.mass_, right.centroid_, &centroid_);
    la::Scale(1.0 / mass_, &centroid_);
    Vector temp;
    la::SubInit(left.centroid_, right.centroid_, &temp);
    radius_ = sqrt(la::Dot(temp, temp));
    radius_ = radius_ + left.radius_ + right.radius_;    
  }

  void UpdateCentroid(double time_step, const Vector& external_vel){
    la::AddExpert(time_step, velocity_, &centroid_);
    la::AddExpert(time_step, external_vel, &centroid_);
  }

  // Update Positions
  void UpdateCentroid(double time_step){    
    la::AddExpert(time_step, velocity_, &centroid_);
  }

  // Apply force
  void ApplyForce(const Vector& force){    
    la::AddExpert(1.0 / mass_, force, &velocity_);
  }

  void GetVelocity(Vector* vel){
    vel->CopyValues(velocity_);
    velocity_.SetZero();
  }

  
};

typedef BinarySpaceTree<DHrectBound<2>, Matrix, ParticleStat> ParticleTree;

#endif




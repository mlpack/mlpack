/**
 * @file atom_tree.h
 *
 * KD-tree stucture for molecular dynamics simulation. 
 * Each node stores a bounding box, centroid, and number of
 * atoms. Leaf nodes also store the velocity of the corresponding 
 * atom. Atoms are assumed to be identical, as this is the
 * largely the case for applications of the LJ potential.
 * 
 */

#include "fastlib/fastlib.h"

struct AtomStat {
  double mass;
  Vector centroid;
  Vector velocity; 

  /**
   * Default Initialization
   */
  void Init(){
    centroid.Init(3);
    velocity.Init(3);
    centroid.SetZero();
    velocity.SetZero();
    mass = 0;
  }

  /**
   * Init funciton for leaf node. Each leaf corresponds to a single atom.
   */
  void Init(const Matrix& dataset, int start, int count){
    centroid.Init(3);
    centroid.SetZero();
    int i;
    Vector temp;      
    mass = count;
    for (i = 0; i < count ; i++){
      dataset.MakeColumnVector(start+i, &temp);
      la::AddTo(temp, &centroid);	
    }
    la::Scale(1.0 / mass, &centroid);   
    velocity.Init(3);
    velocity.SetZero();
  }


  /**
   * Init function to build node from two children, tracking mass and
   * centroid of each node. Since the updating of velocities is done
   * as a single tree search, non-leaves do not need to store velocity.
   */
  void Init(const Matrix& dataset, int start, int count, 
	    const AtomStat &left_stat, const AtomStat &right_stat){
    
    Vector v_r_, v_l_;
    mass = count;
    la::ScaleInit(left_stat.mass,  left_stat.centroid, &v_l_);
    la::ScaleInit(right_stat.mass, right_stat.centroid, &v_r_);  
    la::AddTo(v_r_, &v_l_);
    la::ScaleInit(1.0 / mass, v_l_, &centroid);
    velocity.Init(3);
    velocity.SetZero();
  }

  // Update Leaf node centroid
  void UpdateCentroid(double time_step){
    int i;
    for (i = 0; i < 3; i++){
      centroid[i] = centroid[i] + time_step*velocity[i];
    }
  }

  // Update non-leaf centroid
  void UpdateCentroid(const AtomStat &left_stat, const AtomStat &right_stat){
    int i;
    for (i = 0; i < 3; i++){
      centroid[i] = (left_stat.centroid[i] * left_stat.mass + 
		     right_stat.centroid[i] * right_stat.mass) / mass;
    }
  }

};

typedef BinarySpaceTree<DHrectBound<2>, Matrix, AtomStat> AtomTree;


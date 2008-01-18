/**
 * KD-tree stucture for molecular dynamics simulation. 
 * Each node stores a bounding box, centroid, and number of
 * atoms. Velocity for dynamics problems can be stored in 
 * a separate matrix. 
 *
 * Eventually, we will have to add stats to permit multiple
 * types of atoms in a simulation.
 * 
 * J. Waters 
 * Begun 11-13-2007 
 */

#include "fastlib/fastlib.h"
#include "tree/kdtree.h"
#include "tree/spacetree.h"
#include "tree/bounds.h"


// We need to track total number of atoms and centroid for each node.
struct AtomStat {
  double mass;
  Vector centroid;
  Vector velocity; // At present, only the velocities at leaf nodes
                    // are used and calculated properly.
 
  // Basic Initialization
  void Init(){
    centroid.Init(3);
    velocity.Init(3);
    centroid.SetZero();
    velocity.SetZero();
    mass = 0;
  }

  // Leaf node initialization
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


  // Non-leaf node initialization
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

};

typedef BinarySpaceTree<DHrectBound<2>, Matrix, AtomStat> AtomTree;


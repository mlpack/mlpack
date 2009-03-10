/**
 * @file physics_system.h
 *
 * @author Jim Waters (jwaters6@gatech.edu)
 *
 *
 */


#ifndef MULTI_PHYSICS_SYSTEM_H
#define MULTI_PHYSICS_SYSTEM_H

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


class MultiPhysicsSystem{

  static const int FREE = 0;
  static const int PERIODIC = 1;
  static const int FIXED = 2;

private:
  // Input data set
  Matrix atoms_;
  Matrix forces_;
  Matrix axilrod_teller_;
  Matrix diffusion_;
  // Trees store current state of system
  ParticleTree *system_, *query_;
  ArrayList<int> old_from_new_map_, new_from_old_map_;  

  Vector dimensions_, signs_, powers_;
  
  
  double time_step_, boundary_, virial_, temperature_, cutoff_;
  bool three_body_;
  int n_atoms_, n_trips_;
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
    if (cutoff_ > 0){
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


 
  // Three body force between three nodes
  void ThreeBodyForce_(ParticleTree* a, ParticleTree* b, ParticleTree* c){
    Vector r_ij, r_jk, r_ki;
    la::SubInit(a->stat().centroid_, b->stat().centroid_, &r_ij);
    la::SubInit(b->stat().centroid_, c->stat().centroid_, &r_jk);
    la::SubInit(c->stat().centroid_, a->stat().centroid_, &r_ki);
    if (boundary_ == PERIODIC){
      AdjustVector_(&r_ij);
      AdjustVector_(&r_jk);
      AdjustVector_(&r_ki);
    }    
    double AA, BB, CC, AB, AC, BC, coef1, coef2, coef3;
    double cosines, denom;    
    // Extra Terms
    double denom2, coef1b, coef2b, coef3b;
    AA = la::Dot(r_ij, r_ij);
    CC = la::Dot(r_ki, r_ki);
    BB = la::Dot(r_jk, r_jk);
    AC = la::Dot(r_ij, r_ki);
    AB = la::Dot(r_ij, r_jk);
    BC = la::Dot(r_ki, r_jk);
    cosines = BC*AC*AB;
    denom = AA*BB*CC;
    denom2 = pow(denom, 3.5);
    denom = pow(denom, 2.5);
    denom = 3.0 * a->stat().axilrod_[0] * b->stat().axilrod_[0] *
       c->stat().axilrod_[0] / denom;   
    denom = denom * time_step_;
    Vector force_i, force_j, force_k;
    coef1 = denom*(2.0*AB*AC + BC*BC - 5.0*cosines/AA);
    coef2 = denom*(2.0*BC*AC + AB*AB - 5.0*cosines/CC);
    coef3 = denom*(2.0*AB*BC + AC*AC - 5.0*cosines/BB);	      
    la::ScaleInit(-coef1, r_ij, &force_i);
    la::AddExpert( coef2, r_ki, &force_i);
    la::ScaleInit( coef1, r_ij, &force_j);
    la::AddExpert(-coef3, r_jk, &force_j);
    
    // Extra Term stuff    
    denom2 = 5.0 *  a->stat().axilrod_[1] * b->stat().axilrod_[1] *
      c->stat().axilrod_[1] / denom2;
    
    denom2 = denom2 * time_step_;
    coef1b = denom2*(BC*AA + BC*BC + 3.0*AC*AB - 14.0*cosines/AA);
    coef2b = denom2*(AB*CC + AB*AB + 3.0*AC*BC - 14.0*cosines/CC);
    coef3b = denom2*(AC*BB + AC*AC + 3.0*BC*AB - 14.0*cosines/BB);
    la::AddExpert(-coef1b, r_ij, &force_i);
    la::AddExpert( coef2b, r_ki, &force_i);
    la::AddExpert( coef1b, r_ij, &force_j);
    la::AddExpert(-coef3b, r_jk, &force_j);    
    la::AddInit(force_i, force_j, &force_k);
    la::Scale(-1.0, &force_k);
    //Apply forces  
    a->stat().ApplyForce(force_i);
    b->stat().ApplyForce(force_j);
    c->stat().ApplyForce(force_k);   
    total_triples_++;
  }



  // Three body force between three atoms
  void ThreeBodyForce_(int i, int j, int k){
    Vector pos_i, pos_j, pos_k;   
    Vector r_ij, r_jk, r_ki;
    atoms_.MakeColumnSubvector(i, 0, 3, &pos_i);
    atoms_.MakeColumnSubvector(j, 0, 3, &pos_j);
    atoms_.MakeColumnSubvector(k, 0, 3, &pos_k);
    la::SubInit(pos_i, pos_j, &r_ij);
    la::SubInit(pos_j, pos_k, &r_jk);
    la::SubInit(pos_k, pos_i, &r_ki);
    double AA, BB, CC, AB, AC, BC, coef1, coef2, coef3;
    double cosines, denom;
    
    // Extra Terms
    double denom2, coef1b, coef2b, coef3b;   
    AA = la::Dot(r_ij, r_ij);
    CC = la::Dot(r_ki, r_ki);
    BB = la::Dot(r_jk, r_jk);
    AC = la::Dot(r_ij, r_ki);
    AB = la::Dot(r_ij, r_jk);
    BC = la::Dot(r_ki, r_jk);
    if (cutoff_ > 0){
      if (sqrt(AA) > cutoff_ || sqrt(BB) > cutoff_ || sqrt(CC) > cutoff_){
	return;
      }
      if (sqrt(AA) > 7.0 & sqrt(BB) > 7.0 & sqrt(CC) > 7.0){
	return;
      }
    }
    cosines = BC*AC*AB;
    denom = AA*BB*CC;
    denom2 = pow(denom, 3.5);      
    denom = pow(denom, 2.5);
    denom = 3.0 * axilrod_teller_.get(0,i) * axilrod_teller_.get(0,j) *
      axilrod_teller_.get(0,k) / denom;
       Vector force_i, force_j, force_k;
    coef1 = denom*(2.0*AB*AC + BC*BC - 5.0*cosines/AA);
    coef2 = denom*(2.0*BC*AC + AB*AB - 5.0*cosines/CC);
    coef3 = denom*(2.0*AB*BC + AC*AC - 5.0*cosines/BB);	      
    la::ScaleInit(-coef1, r_ij, &force_i);
    la::AddExpert( coef2, r_ki, &force_i);
    la::ScaleInit( coef1, r_ij, &force_j);
    la::AddExpert(-coef3, r_jk, &force_j);
   
    // Extra Term stuff
       
    denom2 = 5.0 * axilrod_teller_.get(1, i) * axilrod_teller_.get(1, j) *
      axilrod_teller_.get(1, k) / denom2;    
    coef1b = denom2*(BC*AA + BC*BC + 3.0*AC*AB - 14.0*cosines/AA);
    coef2b = denom2*(AB*CC + AB*AB + 3.0*AC*BC - 14.0*cosines/CC);
    coef3b = denom2*(AC*BB + AC*AC + 3.0*BC*AB - 14.0*cosines/BB);
    la::AddExpert(-coef1b, r_ij, &force_i);
    la::AddExpert( coef2b, r_ki, &force_i);
    la::AddExpert( coef1b, r_ij, &force_j);
    la::AddExpert(-coef3b, r_jk, &force_j);      
    la::AddInit(force_i, force_j, &force_k);
    la::Scale(-1.0, &force_k);
    //Apply forces
    pos_i.Destruct();
    pos_j.Destruct();
    pos_k.Destruct();
    atoms_.MakeColumnSubvector(i, 4, 3, &pos_i);
    atoms_.MakeColumnSubvector(j, 4, 3, &pos_j);
    atoms_.MakeColumnSubvector(k, 4, 3, &pos_k); 
    la::AddExpert(time_step_ / atoms_.get(3,i), force_i, &pos_i);
    la::AddExpert(time_step_ / atoms_.get(3,j), force_j, &pos_j);
    la::AddExpert(time_step_ / atoms_.get(3,k), force_k, &pos_k);   
    total_triples_++;
  }
    
  /**
   * Force bounding functions
   */

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
    r_min = max(r_min, 2.8);
    Vector max, min, fmax, fmin;
    max.Init(3);
    min.Init(3);
    fmax.Init(3);
    fmin.Init(3);
    fmax.SetZero();
    fmin.SetZero();
    for(int i = 0; i < 3; i++){
      max[i] = query->bound().MaxDelta(ref->bound(), dimensions_[i],i);
      min[i] = query->bound().MinDelta(ref->bound(), dimensions_[i],i);
    }
    for (int i = 0; i < forces_.n_rows(); i++){
      int power = (int)powers_[i];
      double temp = query->stat().interactions_[i].coef()*
	ref->stat().interactions_[i].coef()*power;      
      
      
      for (int j = 0; j < 3; j++){
	if (max[j]*temp > 0){
	  fmax[j] = fmax[j] + max[j]*temp*pow(r_min, power-2);
	} else {
	  fmax[j] = fmax[j] + max[j]*temp*pow(r_max, power-2);
	}
	if( min[j]*temp > 0){
	  fmin[j] = fmin[j] + min[j]*temp*pow(r_max, power-2);
	} else {
	  fmin[j] = fmin[j] + min[j]*temp*pow(r_min, power-2);
	}
      }
      
    }    
    // Get range from omitting three body interactions 
    double coef = query->stat().axilrod_[0]*ref->stat().axilrod_[0]*
      (query->stat().axilrod_[0] + ref->stat().axilrod_[0]);
    double dist = 2.8;
    for (int j = 0; j < 3; j++){
      if (coef*max[j] > 0){
	fmax[j] = fmax[j] + 0.25*coef*max[j]*(15 / pow(r_min, 11) - 
          48 /pow(r_max,11));  
      } else {
	fmax[j] = fmax[j] + 0.25*coef*max[j]*(15 / pow(r_max, 11) - 
          24*(1.0/(pow(dist, 3)*pow(r_min,8)) + 1.0/(dist*pow(r_min, 10))));  
      }
      if (coef*min[j] > 0){
	fmin[j] = fmin[j] + 0.25*coef*min[j]*(15 / pow(r_min, 11) - 
          48/pow(r_max,11)); 
      } else {
	fmin[j] = fmin[j] + 0.25*coef*min[j]*(15 / pow(r_max, 11) - 
          24*(1.0/(pow(dist, 3)*pow(r_min,8)) + 1.0/(dist*pow(r_min, 10))));  
      }
    }
    
    Vector delta, err;
    la::SubInit(fmax, fmin, &delta);
    err.Init(2);
    err[0] = la::Dot(delta,delta);
    err[1] = err[0] / (ref->stat().mass_*ref->stat().mass_);
    err[0] = err[0] / (query->stat().mass_*query->stat().mass_);
    la::ScaleOverwrite(time_step_*time_step_ , err, bounds);
  }



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
 


 void GetForceRangeTriple_(ParticleTree* i, ParticleTree* j,
		      ParticleTree* k, Vector* bounds){    
    double a_max, b_max, c_max, a_min, b_min, c_min;   
    if (boundary_ == PERIODIC){
      a_min = sqrt(i->bound().PeriodicMinDistanceSq(j->bound(),dimensions_));
      a_max = sqrt(i->bound().PeriodicMaxDistanceSq(j->bound(),dimensions_));
      b_min = sqrt(j->bound().PeriodicMinDistanceSq(k->bound(),dimensions_));
      b_max = sqrt(j->bound().PeriodicMaxDistanceSq(k->bound(),dimensions_));
      c_min = sqrt(k->bound().PeriodicMinDistanceSq(i->bound(),dimensions_));
      c_max = sqrt(k->bound().PeriodicMaxDistanceSq(i->bound(),dimensions_));
    } else {
      a_min = sqrt(i->bound().MinDistanceSq(j->bound()));     
      a_max = sqrt(i->bound().MaxDistanceSq(j->bound()));
      b_min = sqrt(j->bound().MinDistanceSq(k->bound()));     
      b_max = sqrt(j->bound().MaxDistanceSq(k->bound()));
      c_min = sqrt(k->bound().MinDistanceSq(i->bound()));     
      c_max = sqrt(k->bound().MaxDistanceSq(i->bound()));
    }    
    a_min = max(a_min, 2.8);
    b_min = max(b_min, 2.8);
    c_min = max(c_min, 2.8);
    Vector maxA, maxB, maxC, minA, minB, minC;
    minA.Init(3);
    maxA.Init(3);
    minB.Init(3);
    maxB.Init(3);
    minC.Init(3);
    maxC.Init(3);
    for (int d = 0; d < 3; d++){
      maxA[d] = i->bound().MaxDelta(j->bound(), dimensions_[d], d);
      maxB[d] = j->bound().MaxDelta(k->bound(), dimensions_[d], d);
      maxC[d] = k->bound().MaxDelta(i->bound(), dimensions_[d], d);
      minA[d] = i->bound().MinDelta(j->bound(), dimensions_[d], d);
      minB[d] = j->bound().MinDelta(k->bound(), dimensions_[d], d);
      minC[d] = k->bound().MinDelta(i->bound(), dimensions_[d], d);
    }
    Vector fmaxA, fmaxB, fmaxC, fminA, fminB, fminC;
    fminA.Init(3);
    fminB.Init(3); 
    fminC.Init(3);
    fmaxA.Init(3); 
    fmaxB.Init(3);
    fmaxC.Init(3);
    
    double coef;   
    coef = fabs(3.0 * i->stat().axilrod_[0] * j->stat().axilrod_[0] *
      k->stat().axilrod_[0] / 8.0)*time_step_;  
    Vector coef_max, coef_min;
    coef_max.Init(3);
    coef_min.Init(3);
    coef_min[0] = coef*(5*(b_min/pow(c_max,5) + c_min / pow(b_max, 5))
	 / (pow(a_max,7)) - 
	 ((((5.0 / (b_min*pow(c_min,3)) + 5.0/(c_min*pow(b_min,3))) / 
	    (a_min*a_min) + 2.0/pow(b_min*c_min, 3) + 3.0/(b_min*pow(c_min, 5))
	    + 3.0/(c_min*pow(b_min,5)))/(a_min*a_min) + 
	   1.0/(pow(b_min,5)*pow(c_min,3))  + 1.0/(pow(b_min,3)*pow(c_min,5)))
	  /(a_min*a_min) + 1.0 / (pow(b_min, 5)*pow(c_min,5)) / a_min));
    coef_max[0] = coef*(5*(b_max/pow(c_min,5) + c_max/ pow(b_min, 5))
	 / (pow(a_min,7)) - 
	 ((((5.0 / (b_max*pow(c_max,3)) + 5.0/(c_max*pow(b_max,3))) / 
	    (a_max*a_max) + 2.0/pow(b_max*c_max, 3) + 3.0/(b_max*pow(c_max, 5))
	    + 3.0/(c_max*pow(b_max,5)))/(a_max*a_max) + 
	   1.0/(pow(b_max,5)*pow(c_max,3))  + 1.0/(pow(b_max,3)*pow(c_max,5)))
	  /(a_max*a_max) + 1.0 / (pow(b_max, 5)*pow(c_max,5)) / a_max));
    coef_min[1] = coef*(5*(a_min/pow(c_max,5) + c_min / pow(a_max, 5))
	 / (pow(b_max,7)) - 
	 ((((5.0 / (a_min*pow(c_min,3)) + 5.0/(c_min*pow(a_min,3))) / 
	    (b_min*b_min) + 2.0/pow(a_min*c_min, 3) + 3.0/(a_min*pow(c_min, 5))
	    + 3.0/(c_min*pow(a_min,5)))/(b_min*b_min) + 
	   1.0/(pow(a_min,5)*pow(c_min,3))  + 1.0/(pow(a_min,3)*pow(c_min,5)))
	  /(b_min*b_min) + 1.0 / (pow(a_min, 5)*pow(c_min,5)) / b_min));
    coef_max[1] = coef*(5*(a_max/pow(c_min,5) + c_max/ pow(a_min, 5))
	 / (pow(b_min,7)) - 
	 ((((5.0 / (a_max*pow(c_max,3)) + 5.0/(c_max*pow(a_max,3))) / 
	    (b_max*b_max) + 2.0/pow(a_max*c_max, 3) + 3.0/(a_max*pow(c_max, 5))
	    + 3.0/(c_max*pow(a_max,5)))/(b_max*b_max) + 
	   1.0/(pow(a_max,5)*pow(c_max,3))  + 1.0/(pow(a_max,3)*pow(c_max,5)))
	  /(b_max*b_max) + 1.0 / (pow(a_max, 5)*pow(c_max,5)) / b_max));
    coef_min[2] = coef*(5*(a_min/pow(b_max,5) + b_min / pow(a_max, 5))
	 / (pow(c_max,7)) - 
	 ((((5.0 / (a_min*pow(b_min,3)) + 5.0/(b_min*pow(a_min,3))) / 
	    (c_min*c_min) + 2.0/pow(a_min*b_min, 3) + 3.0/(a_min*pow(b_min, 5))
	    + 3.0/(b_min*pow(a_min,5)))/(c_min*c_min) + 
	   1.0/(pow(a_min,5)*pow(b_min,3))  + 1.0/(pow(a_min,3)*pow(b_min,5)))
	  /(c_min*c_min) + 1.0 / (pow(a_min, 5)*pow(b_min,5)) / c_min));
    coef_max[2] = coef*(5*(a_max/pow(b_min,5) + b_max/ pow(a_min, 5))
	 / (pow(c_min,7)) - 
	 ((((5.0 / (a_max*pow(b_max,3)) + 5.0/(b_max*pow(a_max,3))) / 
	    (c_max*c_max) + 2.0/pow(a_max*b_max, 3) + 3.0/(a_max*pow(b_max, 5))
	    + 3.0/(b_max*pow(a_max,5)))/(c_max*c_max) + 
	   1.0/(pow(a_max,5)*pow(b_max,3))  + 1.0/(pow(a_max,3)*pow(b_max,5)))
	  /(c_max*c_max) + 1.0 / (pow(a_max, 5)*pow(b_max,5)) / c_max));
      

    for (int d = 0; d < 3; d++){
      if (coef*minA[d] > 0){
	fminA[d] = minA[d]*coef_min[0];
      } else {
	fminA[d] = minA[d]*coef_max[0];
      }
      if (coef*maxA[d] > 0){
	fmaxA[d] = maxA[d]*coef_max[0];
      } else {
	fmaxA[d] = maxA[d]*coef_min[0];
      }
      if (coef*minB[d] > 0){
	fminB[d] = minB[d]*coef_min[1];
      } else {
	fminB[d] = minB[d]*coef_max[1];
      }
      if (coef*maxB[d] > 0){
	fmaxB[d] = maxB[d]*coef_max[1];
      } else {
	fmaxB[d] = maxB[d]*coef_min[1];
      }
      if (coef*minC[d] > 0){
	fminC[d] = minC[d]*coef_min[2];
      } else {
	fminC[d] = minC[d]*coef_max[2];
      }
      if (coef*maxC[d] > 0){
	fmaxC[d] = maxC[d]*coef_max[2];
      } else {
	fmaxC[d] = maxC[d]*coef_min[2];
      }
    }


    Vector range, fminI, fminJ, fminK, fmaxI, fmaxJ, fmaxK;
    la::SubInit(fmaxA, fminC, &fmaxI); 
    la::SubInit(fmaxB, fminA, &fmaxJ); 
    la::SubInit(fmaxC, fminB, &fmaxK); 
    la::SubInit(fminA, fmaxC, &fminI); 
    la::SubInit(fminB, fmaxA, &fminJ); 
    la::SubInit(fminC, fmaxB, &fminK); 

    range.Init(3);
    range[0] = la::DistanceSqEuclidean(fminI, fmaxI);
    range[1] = la::DistanceSqEuclidean(fminJ, fmaxJ);
    range[2] = la::DistanceSqEuclidean(fminK, fmaxK);
    range[0] = range[0] / (i->stat().mass_*i->stat().mass_);
    range[1] = range[1] / (j->stat().mass_*j->stat().mass_);
    range[2] = range[2] / (k->stat().mass_*k->stat().mass_);
    la::ScaleOverwrite(1.0, range, bounds);   
  }


  int GetForceRangeTripleCutoff_(ParticleTree* i, ParticleTree* j,
				 ParticleTree* k){    
    int result;
    double a_min, b_min, c_min;   
    if (boundary_ == PERIODIC){
      a_min = sqrt(i->bound().PeriodicMinDistanceSq(j->bound(),dimensions_));  
      b_min = sqrt(j->bound().PeriodicMinDistanceSq(k->bound(),dimensions_));  
      c_min = sqrt(k->bound().PeriodicMinDistanceSq(i->bound(),dimensions_));  
    } else {
      a_min = sqrt(i->bound().MinDistanceSq(j->bound()));    
      b_min = sqrt(j->bound().MinDistanceSq(k->bound()));       
      c_min = sqrt(k->bound().MinDistanceSq(i->bound()));    
    }      
    result = (a_min > cutoff_)+(b_min > cutoff_)+(c_min > cutoff_); 
    result = result +(a_min > 7.0)*(b_min > 7.0)*(c_min > 7.0);
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
    // Two-Body only. We can't evalute three body between two nodes
    // without considering each particle.
    TwoBodyForce_(query, ref);
  }

  void EvaluateNodeForcesThree_(ParticleTree* query, ParticleTree* ref1,
				ParticleTree* ref2){
    ThreeBodyForce_(query, ref1, ref2);    
  }

  // These forces will always be between disjoint nodes
  void EvaluateLeafForcesThree_(ParticleTree* query, ParticleTree* ref1,
				ParticleTree* ref2){    
    for(int i = query->begin(); i < query->begin() + query->count(); i++){
      for(int j = ref1->begin(); j < ref1->count() + ref1->begin(); j++){
	for(int k = ref2->begin(); k < ref2->count() + ref2->begin(); k++){
	  ThreeBodyForce_(i,j,k);	  
	}
      }
    }
  }

 
  // This will also cover overlap cases near the diagonal,
  // so query and ref may be the same node. We can evalute three
  // body forces between these two nodes by considering each triple.
  void EvaluateLeafForcesDual_(ParticleTree* query, ParticleTree* ref){
    // Two and Three Body
     for(int i = query->begin(); i < query->begin()+query->count(); i++){
      for(int j = ref->begin(); j < ref->count() + ref->begin(); j++){
	TwoBodyForce_(i,j);	
	for(int k = j+1; k < ref->count() + ref->begin(); k++){
	  ThreeBodyForce_(i,j,k);	  
	}   
	for(int k = i+1; k <query->begin() + query->count(); k++){
	  ThreeBodyForce_(i,j,k);
	}	
      }           
    }
  }

  void EvaluateLeafForcesSame_(ParticleTree* query){
    for (int i = query->begin(); i < query->begin() + query->count(); i++){
      for (int j = i+1; j < query->begin() + query->count(); j++){
	TwoBodyForce_(i,j);
	for (int k = j+1; k < query->begin() + query->count(); k++){
	  ThreeBodyForce_(i,j,k);	  
	}
      }
    }
  }


  // End Force Evaluation Routines.


  /**
   * Momentum updating routines.
   */
  void UpdateMomentumDual_(ParticleTree* query, ParticleTree* ref){
    // if (ref->begin() <= query->begin()){
    //   return;
    // }
    Vector force_range, error_bound;   
    error_bound.Init(2);
    force_range.Init(2);
    GetForceRangeDual_(query, ref, &force_range);
    error_bound[0] = force_bound_*ref->count()*(ref->count()-1.0) 
      / (2.0*n_trips_);
    error_bound[1] = force_bound_*query->count()*(query->count()-1.0) 
      / (2.0*n_trips_);    
    // Can we evaluate force here?
    if (force_range[0] < error_bound[0] & force_range[1] < error_bound[1]){
      EvaluateNodeForcesDual_(query, ref);
    } else {
      // Or do we recurse down further?
      int a,b;
      a = query->count();
      b = ref->count();  
      if (a >= b & a > leaf_size_){
	UpdateMomentumDual_(query->left(), ref);
	UpdateMomentumDual_(query->right(), ref);
	UpdateMomentumThree_(query->left(), query->right(), ref);
      } else {
	if (b > leaf_size_){
	  UpdateMomentumDual_(query, ref->left());
	  UpdateMomentumDual_(query, ref->right());
	  UpdateMomentumThree_(query, ref->left(), ref->right());
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
 
  
  void UpdateMomentumThree_(ParticleTree* query, ParticleTree* ref1, 
			    ParticleTree* ref2){
    // Avoid double counting any interactions.
    //   if (ref1->begin() < query->begin() || ref2->begin() < ref1->begin()){
    //     return;
    //    }

    Vector force_range, error_bound;   
    error_bound.Init(3);
    force_range.Init(3);
    GetForceRangeTriple_(query, ref1, ref2, &force_range);
    error_bound[0] = force_bound_*ref1->count()*ref2->count()/(2.0*n_trips_);
    error_bound[1] = force_bound_*query->count()*ref2->count()/(2.0*n_trips_);
    error_bound[2] = force_bound_*ref1->count()*query->count()/(2.0*n_trips_);
    // Can we evaluate force here?
    if (force_range[0] < error_bound[0] & force_range[1] < error_bound[1] &
	force_range[2] < error_bound[2]){
      EvaluateNodeForcesThree_(query, ref1, ref2);
    } else {
      // Or do we recurse down further?
      int a,b,c;
      a = query->count();
      b = ref1->count();
      c = ref2->count();
      if (a >= b & a >= c & a > leaf_size_){
	UpdateMomentumThree_(query->left(), ref1, ref2);
	UpdateMomentumThree_(query->right(), ref1, ref2);
      } else {
	if (b >= c & b > leaf_size_){
	  UpdateMomentumThree_(query, ref1->left(), ref2);
	  UpdateMomentumThree_(query, ref1->right(), ref2);
	} else {
	  if (c > leaf_size_){
	    UpdateMomentumThree_(query, ref1, ref2->left());
	    UpdateMomentumThree_(query, ref1, ref2->right());
	  } else {
	    // Base Case
	    EvaluateLeafForcesThree_(query, ref1, ref2);
	  }
	} 
      }
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
	UpdateMomentumThreeCutoff_(query->left(), query->right(), ref);
      } else {
	if (b > leaf_size_){
	  UpdateMomentumDualCutoff_(query, ref->left());
	  UpdateMomentumDualCutoff_(query, ref->right());
	  UpdateMomentumThreeCutoff_(query, ref->left(), ref->right());
	} else {
	  // Base Case	  
	  EvaluateLeafForcesDual_(query, ref);	   	  
	}	 
      }
    }
  }


  void UpdateMomentumThreeCutoff_(ParticleTree* query, ParticleTree* ref1, 
				  ParticleTree* ref2){   
    int prune = GetForceRangeTripleCutoff_(query, ref1, ref2);
    if (!prune) {
      // Or do we recurse down further?
      int a,b,c;
      a = query->count();
      b = ref1->count();
      c = ref2->count();
      if (a >= b & a >= c & a > leaf_size_){
	UpdateMomentumThreeCutoff_(query->left(), ref1, ref2);
	UpdateMomentumThreeCutoff_(query->right(), ref1, ref2);
      } else {
	if (b >= c & b > leaf_size_){
	  UpdateMomentumThreeCutoff_(query, ref1->left(), ref2);
	  UpdateMomentumThreeCutoff_(query, ref1->right(), ref2);
	} else {
	  if (c > leaf_size_){
	    UpdateMomentumThreeCutoff_(query, ref1, ref2->left());
	    UpdateMomentumThreeCutoff_(query, ref1, ref2->right());
	  } else {
	    // Base Case
	    EvaluateLeafForcesThree_(query, ref1, ref2);	    
	  }
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
      node->bound() |= node->left()->bound();
      node->bound() |= node->right()->bound();      
    } else {  // Base Case                
      node->bound().Reset();     
      la::AddTo(node->stat().velocity_, vel); 
      for(int i = node->begin(); i < node->begin() + node->count(); i++){
	Vector pos, temp;
	atoms_.MakeColumnSubvector(i, 4, 3, &temp);
	la::AddTo(*vel, &temp);
	atoms_.MakeColumnSubvector(i, 0, 3, &pos);
	la::AddExpert(time_step_, temp, &pos);
	node->bound() |= pos;
      }            
      node->stat().InitKinematics(node->begin(), node->count(), atoms_);
    }      
  }
    


  void InitializeATStats_(ParticleTree* node){
    if (!node->is_leaf()){
      InitializeATStats_(node->left());
      InitializeATStats_(node->right());
      node->stat().MergeAT(node->left()->stat(), node->right()->stat());
    } else {
      node->stat().InitAT(&axilrod_teller_);
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

    FORBID_ACCIDENTAL_COPIES(MultiPhysicsSystem);

public:
     
    MultiPhysicsSystem(){
      system_ = NULL;
      query_ = NULL;
    }
    
    
    ~MultiPhysicsSystem(){     
      if (system_ != NULL){
	delete system_;
      }     
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
    n_trips_ = n_atoms_*(n_atoms_-1)*(n_atoms_-2) / 6;
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

  void InitAxilrodTeller(const Matrix& stats_in){
    // Reindex cols of stats matrix.
    if (system_ != NULL){
      axilrod_teller_.Init(2, n_atoms_);
      for (int i = 0; i < n_atoms_; i++){
	int k = old_from_new_map_[i];
	for (int j = 0; j < 2; j++){
	  axilrod_teller_.set(j, k, stats_in.get(j, i));
	}		
      }
      InitializeATStats_(system_);
    } else {
      axilrod_teller_.Copy(stats_in);
    }
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
    UpdatePositionsRecursion_(system_, &temp);     
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
      Vector pos;
      atoms_.MakeColumnSubvector(i, 0, 3, &pos);
      AdjustVector_(&pos);
      for (int j = 0; j < 3; j++){
	double temp;
	temp = ceil((pos[j] - atoms_.get(j,i)) / dimensions_[j]);
	diffusion_.set(j,i,temp+diffusion_.get(j,i));
	atoms_.set(j,i,pos[j]);	
      }
    }

    system_ = tree::MakeKdTreeMidpointSelective<ParticleTree>(atoms_, dims, 
                leaf_size_, &temp_new_old, &temp_old_new);
       
    for (int i = 0; i < n_atoms_; i++){
      for (int j = 0; j < 3; j++){
	double temp = diffusion_.get(j, temp_old_new[i]);
	diffusion_.set(j, temp_old_new[i], diffusion_.get(j,i));
	diffusion_.set(j, i, temp);
      }
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
    total_triples_ = 0;
    range_evals_ = 0;
    max_force_ = 0;
    virial_ = 0;    
    time_step_ = time_step_in;
    percent_pruned_ = 0;   
    UpdateMomentumMain_(system_);    
    percent_pruned_ = 1.0-2*percent_pruned_ / (n_atoms_*n_atoms_ - n_atoms_); 
    //  printf("Total Computed Triples: %d \n", total_triples_);
    //  printf("Range Evaluations: %d \n", range_evals_);
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
  void CompareToNaive(MultiPhysicsSystem* comp_sys){  
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
      Vector temp1, temp2, temp3;
      int j = i;
      if (system_ != NULL){
	j = old_from_new_map_[j];
      }
      atoms_.MakeColumnSubvector(j, 0, 3, &temp1);
      for (int k = 0; k < 3; k++){
	temp1[k] = temp1[k] + dimensions_[k]*diffusion_.get(k,j);
      }
      old_positions.MakeColumnSubvector(i, 0, 3, &temp2);
      la::SubInit(temp1, temp2, &temp3);     
      diff = diff + la::Dot(temp3, temp3);
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

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
#include "force_error.h"
#include "raddist.h"
#include "../thor_md/periodic_tree.h"
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
  static const int CUTOFF = 0;
  static const int POTENTIAL = 1;
  static const int FORCE = 2;

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
  Vector power3a_, power3b_, power3c_;
  Vector signs3_;
  
  double time_step_, virial_, temperature_, cutoff_, cutoff3_;
  bool three_body_;
  int n_atoms_, boundary_, prune_;
  double n_trips_;
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
    if (prune_ == CUTOFF){
      if (sqrt(AA) > cutoff_ || sqrt(BB) > cutoff_ || sqrt(CC) > cutoff_){
	return;
      }
      if (sqrt(AA) > cutoff3_ & sqrt(BB) > cutoff3_ & sqrt(CC) > cutoff3_){
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
    double rad = sqrt(la::Dot(node_r, node_r)) / Rad;
    rnorm = rnorm / Rnorm;
  
    for (int i = 0; i < forces_.n_rows(); i++){
      int power = abs((int)powers_[i]);  
      double coef = fabs(ref->stat().interactions_[i].coef()*
			 query->stat().interactions_[i].coef()*
			 GetForceTerm_(Rad, rad, Rnorm, rnorm, power));
      range_q = range_q + coef;
      range_r = range_r + coef;
    }    

    double rmin = 2.8;
    double Rmin = sqrt(prdc::MinDistanceSqWrap(ref->bound(), query->bound(),
					       dimensions_));
    for (int d = 0; d < 10; d++){
      int a = abs((int)power3a_[d]);  
      int b = abs((int)power3b_[d]);  
      int c = abs((int)power3c_[d]);  
      double coef = fabs(ref->stat().axilrod_[0]*query->stat().axilrod_[0]*
			 a*signs3_[d]);
      range_q += coef*(Rnorm*ref->stat().axilrod_[0]/(pow(Rmin, a+b+2)*
	pow(rmin,c))+query->stat().axilrod_[0]/(pow(Rmin, b+c)*pow(rmin,a+1)));
      range_r += coef*(Rnorm*query->stat().axilrod_[0]/(pow(Rmin, a+b+2)*
        pow(rmin,c))+ref->stat().axilrod_[0]/(pow(Rmin, b+c)*pow(rmin, a+1)));
      
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

    double rmin = 2.8;
    double Rmin = sqrt(prdc::MinDistanceSqWrap(ref->bound(), 
					       query->bound(), dimensions_));
    coef = fabs(4*ref->stat().axilrod_[0]*query->stat().axilrod_[0] / 
		(pow(rmin, 3)*pow(Rmin, 6)));
    range_q = range_q + coef*(ref->stat().axilrod_[0] + query->stat().axilrod_[0]/2.0);
    range_r = range_r + coef*(query->stat().axilrod_[0] + ref->stat().axilrod_[0]/2.0);
    range_q = range_q / query->count();
    range_r = range_r / ref->count();

    Vector err; 
    err.Init(2);
    err[0] = fabs(range_q);
    err[1] = fabs(range_r);
  
    la::ScaleOverwrite(time_step_, err, bounds);
  }

   void GetForceRangeTriple_(ParticleTree* i, ParticleTree* j, ParticleTree *k,
			  Vector* bounds){       
    double range_i = 0, range_j = 0, range_k = 0;
    double Rij, Rjk, Rki, rij, rki, rjk, ri, rj, rk;
    double rnij = 0, rnjk = 0, rnki = 0, Rnij = 0, Rnjk = 0, Rnki = 0;
    Vector delta_ij, delta_jk, delta_ki;
    la::SubInit(i->stat().centroid_, j->stat().centroid_, &delta_ij);
    la::SubInit(j->stat().centroid_, k->stat().centroid_, &delta_jk);
    la::SubInit(k->stat().centroid_, i->stat().centroid_, &delta_ki);
    AdjustVector_(&delta_ij);
    AdjustVector_(&delta_jk);
    AdjustVector_(&delta_ki);
    Rij = sqrt(la::Dot(delta_ij, delta_ij));
    Rjk = sqrt(la::Dot(delta_jk, delta_jk));
    Rki = sqrt(la::Dot(delta_ki, delta_ki));

    Vector bi, bj, bk;
    bi.Init(3);
    bj.Init(3);
    bk.Init(3);
    for (int d = 0; d < 3; d++){
      bi[d] = i->bound().width(d, dimensions_[d]) / 2;
      bj[d] = j->bound().width(d, dimensions_[d]) / 2;
      bk[d] = k->bound().width(d, dimensions_[d]) / 2;
      rnij = rnij + bi[d] + bj[d];
      rnjk = rnjk + bj[d] + bk[d];
      rnki = rnki + bk[d] + bi[d];
      Rnij = Rnij + fabs(delta_ij[d]);
      Rnjk = Rnjk + fabs(delta_jk[d]);
      Rnki = Rnki + fabs(delta_ki[d]);
    }
    ri = la::Dot(bi, bi);
    rj = la::Dot(bj, bj);
    rk = la::Dot(bk, bk);
    rij = sqrt(ri+rj)/Rij;
    rjk = sqrt(rj+rk)/Rjk;
    rki = sqrt(rk+ri)/Rki;
  
    for (int d = 0; d < 10; d++){
      int a = abs((int)power3a_[d]);  
      int b = abs((int)power3b_[d]);  
      int c = abs((int)power3c_[d]);  
      double coef = i->stat().axilrod_[0]*j->stat().axilrod_[0]*
	k->stat().axilrod_[0]*signs3_[d];
      range_i += coef*GetPotentialTermPt_(Rjk, rjk, b)*
	(GetForceTerm_(Rij, rij, Rnij, rnij,a)*GetPotentialTerm_(Rki, rki,c)+
	 GetForceTerm_(Rki, rki, Rnki, rnki,c)*GetPotentialTerm_(Rij, rij,a));
      range_j+= coef*GetPotentialTermPt_(Rki, rki, c)*
	(GetForceTerm_(Rij, rij, Rnij, rnij,a)*GetPotentialTerm_(Rjk, rjk,b)+
	 GetForceTerm_(Rjk, rjk, Rnjk, rnjk,b)*GetPotentialTerm_(Rij, rij,a));
      range_k+= coef*GetPotentialTermPt_(Rij, rij, a)*
	(GetForceTerm_(Rjk, rjk, Rnjk, rnjk,b)*GetPotentialTerm_(Rki, rki,c)+
	 GetForceTerm_(Rki, rki, Rnki, rnki,c)*GetPotentialTerm_(Rjk, rjk,b)); 
    }
    range_i = range_i / i->count();
    range_j = range_j / j->count();
    range_k = range_k / k->count();
    Vector err;
    err.Init(3);
   
    err[0] = fabs(range_i);
    err[1] = fabs(range_j);
    err[2] = fabs(range_k);
    la::ScaleOverwrite(time_step_, err, bounds);
  }

  void GetPotentialRangeTriple_(ParticleTree* i, ParticleTree* j, ParticleTree *k,
			  Vector* bounds){       
    double range_i = 0, range_j = 0, range_k = 0;
    double Rij, Rjk, Rki, rij, rki, rjk;

    Vector delta_ij, delta_jk, delta_ki;
    la::SubInit(i->stat().centroid_, j->stat().centroid_, &delta_ij);
    la::SubInit(j->stat().centroid_, k->stat().centroid_, &delta_jk);
    la::SubInit(k->stat().centroid_, i->stat().centroid_, &delta_ki);
    AdjustVector_(&delta_ij);
    AdjustVector_(&delta_jk);
    AdjustVector_(&delta_ki);
    Rij = sqrt(la::Dot(delta_ij, delta_ij));
    Rjk = sqrt(la::Dot(delta_jk, delta_jk));
    Rki = sqrt(la::Dot(delta_ki, delta_ki));

    Vector bi, bj, bk;
    bi.Init(3);
    bj.Init(3);
    bk.Init(3);
    for (int d = 0; d < 3; d++){
      bi[d] = i->bound().width(d, dimensions_[d]) / 2;
      bj[d] = j->bound().width(d, dimensions_[d]) / 2;
      bk[d] = k->bound().width(d, dimensions_[d]) / 2; 
    }
    la::AddOverwrite(bi, bj, &delta_ij);
    la::AddOverwrite(bj, bk, &delta_jk);
    la::AddOverwrite(bk, bi, &delta_ki);
    rij = sqrt(la::Dot(delta_ij, delta_ij))/Rij;
    rjk = sqrt(la::Dot(delta_jk, delta_jk))/Rjk;
    rki = sqrt(la::Dot(delta_ki, delta_ki))/Rki;
    delta_ij[0] =   GetPotentialTerm_(Rij, rij, 3);
    delta_ij[1] = GetPotentialTermPt_(Rij, rij, 3);
    delta_jk[0] =   GetPotentialTerm_(Rjk, rjk, 3);
    delta_jk[1] = GetPotentialTermPt_(Rjk, rjk, 3);
    delta_ki[0] =   GetPotentialTerm_(Rki, rki, 3);
    delta_ki[1] = GetPotentialTermPt_(Rki, rki, 3);


    double coef = i->stat().axilrod_[0]*j->stat().axilrod_[0]*
      k->stat().axilrod_[0]*6;
    range_i = coef*delta_ij[0]*delta_ki[0]*delta_jk[1];
    range_j = coef*delta_ij[0]*delta_ki[1]*delta_jk[0];
    range_k = coef*delta_ij[1]*delta_ki[0]*delta_jk[0];
  
    range_i = range_i / i->count();
    range_j = range_j / j->count();
    range_k = range_k / k->count();
    Vector err;
    err.Init(3);
   
    err[0] = fabs(range_i);
    err[1] = fabs(range_j);
    err[2] = fabs(range_k);
    la::ScaleOverwrite(time_step_, err, bounds);
  }

 
  /**
   * Routines for calling force evaluations
   */
  
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



  void SplitDual_(ParticleTree* query, ParticleTree* ref, ForceError* err_q,
		  ForceError* err_r){
    ForceError err_q2;
    err_q2.Copy(err_q);    
    double d1, d2;
    if (boundary_ == PERIODIC){
      d1 = prdc::MinDistanceSqWrap(ref->bound(), query->left()->bound(), 
				   dimensions_);
      d2 = prdc::MinDistanceSqWrap(ref->bound(), query->right()->bound(), 
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
    UpdateMomentumThree_(query->left(), query->right(), ref,
			 err_q, &err_q2, err_r);
    err_q->Merge(err_q2);
  }
  
  void SplitThree_(ParticleTree* query, ParticleTree* ref1, ParticleTree* ref2,
		  ForceError* err_q, ForceError* err_r1, ForceError* err_r2){
    ForceError err_q2;
    err_q2.Copy(err_q);
    double d1, d2;
    if (boundary_ == PERIODIC){
      d1 = prdc::MinDistanceSqWrap(ref1->bound(), query->left()->bound(),dimensions_)*
	prdc::MinDistanceSqWrap(ref2->bound(), query->left()->bound(),dimensions_);
      d2 = prdc::MinDistanceSqWrap(ref1->bound(), query->right()->bound(),dimensions_)*
	prdc::MinDistanceSqWrap(ref2->bound(), query->right()->bound(),dimensions_);
    } else {
      d1 = ref1->bound().MinDistanceSq(query->left()->bound())*
	ref2->bound().MinDistanceSq(query->left()->bound());
      d2 = ref1->bound().MinDistanceSq(query->right()->bound())*
	ref2->bound().MinDistanceSq(query->right()->bound());  
    }
    if (d1 > d2){    
      UpdateMomentumThree_(query->left(), ref1, ref2, err_q, err_r1, err_r2);
      UpdateMomentumThree_(query->right(),ref1, ref2, &err_q2, err_r1, err_r2);
    } else {    
      UpdateMomentumThree_(query->right(),ref1, ref2, &err_q2, err_r1, err_r2);
      UpdateMomentumThree_(query->left(), ref1, ref2, err_q, err_r1, err_r2);
    }       
    err_q->Merge(err_q2);
  }



  int GetPrune_(ParticleTree* i, ParticleTree* j, ParticleTree* k, 
		ForceError* err_i, ForceError* err_j , ForceError* err_k){
    int result = 0;
    if (prune_ == CUTOFF){
      double a_min, b_min, c_min;   
      if (boundary_ == PERIODIC){
	a_min = sqrt(prdc::MinDistanceSqWrap(i->bound(), j->bound(),dimensions_));  
	b_min = sqrt(prdc::MinDistanceSqWrap(j->bound(), k->bound(),dimensions_));  
	c_min = sqrt(prdc::MinDistanceSqWrap(k->bound(), i->bound(),dimensions_));  
      } else {
	a_min = sqrt(i->bound().MinDistanceSq(j->bound()));    
	b_min = sqrt(j->bound().MinDistanceSq(k->bound()));       
	c_min = sqrt(k->bound().MinDistanceSq(i->bound()));    
      }      
      result = (a_min > cutoff_)+(b_min > cutoff_)+(c_min > cutoff_); 
      result = result + (a_min > cutoff3_)*(b_min > cutoff3_)*(c_min > cutoff3_);
    } else {
      Vector range;
      range.Init(3);
      int c1, c2, c3;
      c1 = j->count() * k->count();
      c2 = k->count() * i->count();
      c3 = i->count() * j->count();
      if (prune_ == POTENTIAL){
	GetPotentialRangeTriple_(i,j,k, &range);
      } else {
	GetForceRangeTriple_(i,j,k, &range);
      }
      result = err_i->Check(range[0], c1) * err_j->Check(range[1], c2) * 
	err_k->Check(range[2], c3);
      if (result > 0){
	ThreeBodyForce_(i, j, k);
	err_i->AddVisited(range[0], c1);
	err_j->AddVisited(range[1], c2);
	err_k->AddVisited(range[2], c3);
      }      
    }    
    return result;
  }

 
  int GetPrune_(ParticleTree* i, ParticleTree* j,
		ForceError* err_i, ForceError* err_j){
    int result = 0;
    if (prune_ == CUTOFF){
      double a_min;   
      if (boundary_ == PERIODIC){
	a_min = sqrt(prdc::MinDistanceSqWrap(i->bound(), j->bound(),dimensions_));
      } else {
	a_min = sqrt(i->bound().MinDistanceSq(j->bound()));   
      }      
      result = (a_min > cutoff_);     
    } else {
      Vector range;
      range.Init(2);
      int c1, c2;
      c1 = j->count() * (i->count()-1 + (j->count()-1)/2);
      c2 = i->count() * (j->count()-1 + (i->count()-1)/2);     
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
      if (a >= b & !(query->is_leaf())){
	SplitDual_(query, ref, err_q, err_r);	
      } else {
	if (!(ref->is_leaf())){
	  SplitDual_(ref, query, err_r, err_q);	
	} else {	  
	  // Base Case
	  EvaluateLeafForcesDual_(query, ref);
	  // Update Error Terms
	  int c1, c2;
	  c1 = ref->count() * (query->count()-1 + (ref->count()-1)/2);
	  c2 = query->count() * (ref->count()-1 + (query->count()-1)/2);     
	  err_r->AddVisited(0, c2);
	  err_q->AddVisited(0, c1);	  
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
      err_q->AddVisited(0, (query->count()-1.0)*(query->count()-2.0)/2.0);
    }
  }
 

  void UpdateMomentumThree_(ParticleTree* query, ParticleTree* ref1, 
			    ParticleTree* ref2, ForceError* err_q,
			    ForceError* err_r1, ForceError* err_r2){
    if (GetPrune_(query, ref1, ref2, err_q, err_r1, err_r2) == 0) {
      // Or do we recurse down further?
      int a,b,c;
      a = query->count();
      b = ref1->count();
      c = ref2->count();
      if (a >= b & a >= c & !(query->is_leaf())){
	SplitThree_(query, ref1, ref2, err_q, err_r1, err_r2);
      } else {
	if (b >= c & !(ref1->is_leaf())){
	  SplitThree_(ref1, query, ref2, err_r1, err_q, err_r2);
	} else {
	  if (!(ref2->is_leaf())){
	    SplitThree_(ref2, query, ref1, err_r2, err_q, err_r1);
	  } else {
	    // Base Case
	    int c1, c2, c3;      
	    c1 = ref1->count()*ref2->count();
	    c2 = query->count()*ref2->count();
	    c3 = query->count()*ref1->count();   
	    
	    EvaluateLeafForcesThree_(query, ref1, ref2);
	    err_q->AddVisited(0, c1);
	    err_r1->AddVisited(0, c2);
	    err_r2->AddVisited(0, c3);	    	  
	  }
	} 
      }
    }
  }  
 
 
  void UpdateMomentumNaive_(){
    for (int i = 0; i < n_atoms_; i++){
      for (int j = i+1; j < n_atoms_; j++){
	TwoBodyForce_(i,j);
	for(int k = j+1; k < n_atoms_; k++){
	  ThreeBodyForce_(i, j, k); 	  
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
      r_min = sqrt(prdc::MinDistanceSqWrap(query->bound(), ref->bound(),
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
      query_ = NULL;
    }
    
    
    ~MultiPhysicsSystem(){      
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
    cutoff3_ = pow(cutoff_, 4)*pow(2.7, 5);
    cutoff3_ = pow(10*cutoff3_, 1.0/9.0);
    printf("Cutoff 3: %f \n", cutoff3_);
    n_trips_ = n_atoms_-1;
    n_trips_ = n_trips_*(n_atoms_-2) / 2;
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
    //   system_ = NULL; 
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
    if (&system_ != NULL){     
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

  void InitAxilrodTeller(const Matrix& stats_in, const Matrix& powers){
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
    powers.MakeColumnVector(0, &signs3_);
    powers.MakeColumnVector(1, &power3a_);
    powers.MakeColumnVector(2, &power3b_);
    powers.MakeColumnVector(3, &power3c_); 
  }

  void ReinitAxilrodTeller(const Matrix& stats_in, const Matrix& powers){
    // Reindex cols of stats matrix.
    if (system_ != NULL){    
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
    if (system_ != NULL){
      UpdatePositionsRecursion_(system_, &temp);     
    } else {
      UpdatePositionsNaive_();
    }    
  
  }     


  void RebuildTree(){ 
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
      error_main.Init(force_bound_, n_trips_);
      UpdateMomentumMain_(system_, &error_main);    
      printf("Pairs: %4.0f Triples: %d \n", percent_pruned_, total_triples_);
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

#ifndef TWO_BODY_H
#define TWO_BODY_H

#include "fastlib/fastlib.h"

template<typename QNode, typename RNode, typename TPoint>
class TwoBody{ 

 private:
 
  Vector powers_, signs_;

 static double GetForceTerm_(double R, double r, double Rnorm, double rnorm,
			    int nu){
    double result  = Rnorm*(pow(1-r, -nu-1) -1) / pow(R, nu+2);
    result = result + nu*rnorm / ((nu+2)*pow(R*(1-r), nu+2));
    return result;
  }

   static double GetForceTermPt_(double R, double r, double Rnorm, 
				 double rnorm, int nu){   
    double result  = Rnorm*(pow(1-r, -nu-1) -1 -(nu+1)*r) / pow(R, nu+2);
    result = result + nu*rnorm / ((nu+2)*pow(R, nu+2))*(1/pow(1-r, nu+2)-1); 
    return result;
  }
  

  static double GetPotentialTerm_(double R, double r, int nu){
    double result;
    if (unlikely(r >= 1)){
      return BIG_BAD_NUMBER;
    } else {
      result = (1 / pow(1-r, nu)) /(nu*pow(R, nu));
      return result;
    }
  }

  static double GetPotentialTermPt_(double R, double r, int nu){
    double result;
    if (unlikely(r >= 1)){
      return BIG_BAD_NUMBER;
    } else {
      result = (1 / pow(1-r, nu) - nu*r) /(nu*pow(R, nu));
      return result;
    }
  }


  void AdjustVector_(Vector* vector_in, const Vector& dimensions_) const{
    for(int i = 0; i < 3; i++){
      (*vector_in)[i] = (*vector_in)[i] - dimensions_[i]*
	floor((*vector_in)[i] / dimensions_[i] +0.5);
    }
  }

 
 public:

  OT_DEF(TwoBody){
    OT_MY_OBJECT(signs_);
    OT_MY_OBJECT(powers_);
  }

 public:
  

  void Init(Matrix& params){       
    // Copy columns into powers & signs
    int n;
    n = params.n_rows();
    powers_.Init(n);
    signs_.Init(n);
    for (int i = 0; i < n; i++){
      powers_[i] = params.get(i,0);
      signs_[i] = params.get(i,1);
    }
  } 

  double ForceRange(const TPoint& query, const RNode& ref, const Vector& box)
    const{
    double range_q = 0;
    double  rnorm = 0, Rnorm = 0;
    Vector delta;
    la::SubInit(query.pos_, ref.stat().centroid_, &delta);
    if (box.length() == query.pos_.length()){
      AdjustVector_(&delta, box);
    }
    double Rad = sqrt(la::Dot(delta, delta));
    
    Vector node_r;
    node_r.Init(3);
    for (int i = 0; i < 3; i++){
      node_r[i] = ref.bound().width(i, box[i])/ 2;
      rnorm = rnorm + node_r[i];   
      Rnorm = Rnorm + fabs(delta[i]);
    }
    double rad = sqrt(la::Dot(node_r, node_r)) / Rad;
    rnorm = rnorm / Rnorm;
    
    for (int i = 0; i < powers_.length(); i++){  
      double coef = fabs(ref.stat().coefs_[i]*query.coefs_[i]*
	 signs_[i]*GetForceTermPt_(Rad, rad, Rnorm, rnorm, -(int)powers_[i]));
      range_q = range_q + coef;     
    }       
    return range_q;
  }

  double ForceRange(const QNode& query, const RNode& ref, const Vector& box)
    const{
    double range_q = 0;
    double  rnorm = 0, Rnorm = 0;
    Vector delta;
    la::SubInit(query.stat().centroid_, ref.stat().centroid_, &delta);
    if (box.length() == query.stat().centroid_.length()){
      AdjustVector_(&delta, box);
    }
    double Rad = sqrt(la::Dot(delta, delta));
    
    Vector node_r;
    node_r.Init(3);
    for (int i = 0; i < 3; i++){
      node_r[i] = (query.bound().width(i, box[i]) + 
		   ref.bound().width(i, box[i]))/ 2;
      rnorm = rnorm + node_r[i];   
      Rnorm = Rnorm + fabs(delta[i]);
    }
    double rad = sqrt(la::Dot(node_r, node_r)) / Rad;
    rnorm = rnorm / Rnorm;
    
    for (int i = 0; i < powers_.length(); i++){  
      double coef = fabs(ref.stat().coefs_[i]*query.stat().coefs_[i]*
	 signs_[i]*GetForceTerm_(Rad, rad, Rnorm, rnorm, -(int)powers_[i]));
      range_q = range_q + coef;     
    }    
    range_q = range_q / query.count();  
    return range_q;
  }


  double PotentialRange(const TPoint& query, const RNode& ref,
			const Vector& box) const {
    double range_q = 0;   
    Vector delta;
    la::SubInit(query.pos_, ref.stat().centroid_, &delta);
    if (box.length() == query.pos_.length()){
      AdjustVector_(&delta, box);
    }
    double Rad = sqrt(la::Dot(delta, delta));
    
    Vector node_r;
    node_r.Init(3);
    for (int i = 0; i < 3; i++){
      node_r[i] = ref.bound().width(i, box[i])/ 2;     
    }
    double rad = sqrt(la::Dot(node_r, node_r)) / Rad;
    
    double coef;
    for (int i = 0; i < powers_.length(); i++){
       coef = fabs(ref.stat().coefs_[i]*query.coefs_[i]*
		   signs_[i]*GetPotentialTermPt_(Rad, rad, -(int)powers_[i]));
       range_q = range_q + coef;     
    }      
    return range_q;
  }


  double PotentialRange(const QNode& query, const RNode& ref,
			const Vector& box) const {
    double range_q = 0;   
    Vector delta;
    la::SubInit(query.stat().centroid_, ref.stat().centroid_, &delta);
    if (box.length() == query.stat().centroid_.length()){
      AdjustVector_(&delta, box);
    }
    double Rad = sqrt(la::Dot(delta, delta));
    
    Vector node_r;
    node_r.Init(3);
    for (int i = 0; i < 3; i++){
      node_r[i] = (query.bound().width(i, box[i]) + 
		   ref.bound().width(i, box[i]))/ 2;     
    }
    double rad = sqrt(la::Dot(node_r, node_r)) / Rad;
    
    double coef;
    for (int i = 0; i < powers_.length(); i++){
       coef = fabs(ref.stat().coefs_[i]*query.stat().coefs_[i]*
		   signs_[i]*GetPotentialTerm_(Rad, rad, -(int)powers_[i]));
       range_q = range_q + coef;     
    }    
    range_q = range_q / query.count();
    return range_q;
  }
  

  // Node-Node force vector. Used when potential or force based
  // pruning makes extrinsic prune above leaf level.
  void ForceVector(const QNode& q, const RNode& r, 
		   const Vector& box, Vector* force_out) const{    
    la::SubInit(r.stat().centroid_, q.stat().centroid_, force_out);
    if (box.length() == q.stat().centroid_.length()){
      AdjustVector_(force_out, box);
    }
    double dist = sqrt(la::Dot(*force_out, *force_out));  
    double coef = 0, temp;
    for (int i  = 0; i < powers_.length(); i++){
      temp = -q.stat().coefs_[i]*r.stat().coefs_[i];
      coef = coef + signs_[i]*temp*powers_[i]*pow(dist, powers_[i]-2);
    }             
    la::Scale(coef / q.stat().mass_, force_out);    
  }


  // Node-point force vector. Used when potential or force based pruning
  // makes prune in PairVisitor class.
  void ForceVector(const TPoint& q, const RNode& r, 
		   const Vector& box, Vector* force_out) const{       
    la::SubInit(r.stat().centroid_, q.pos_, force_out);
    if (box.length() == q.pos_.length()){
      AdjustVector_(force_out, box);
    }
    double dist = sqrt(la::Dot(*force_out, *force_out));  
    double coef = 0, temp;
    for (int i  = 0; i < powers_.length(); i++){
      temp = -q.coefs_[i]*r.stat().coefs_[i];
      coef = coef + signs_[i]*temp*powers_[i]*pow(dist, powers_[i]-2);
    }             
    la::Scale(coef /q.mass_, force_out);    
  }

  // Point-point force vector. Used by force or potential pruning
  // when we recurse all the way to leaf level.
  void ForceVector(const TPoint& q, const TPoint& r, 
		   const Vector& box, Vector* force_out) const{    
    la::SubInit(r.pos_, q.pos_, force_out);
    if (box.length() == q.pos_.length()){
      AdjustVector_(force_out, box);
    }
    double dist = sqrt(la::Dot(*force_out, *force_out));  
    double coef = 0, temp;
    for (int i  = 0; i < powers_.length(); i++){
      temp = -q.coefs_[i]*r.coefs_[i];
      coef = coef + signs_[i]*temp*powers_[i]*pow(dist, powers_[i]-2);
    }             
    la::Scale(coef / q.mass_, force_out);    
    
  }

  // Point-point force vector. This may still return zero, if its outside
  // cutoff distance. This is the only force vector function used by cutoff
  // pruning.
  int ForceVector(const TPoint& q, const TPoint& r, const Vector& box, 
		   double cutoff, Vector* force_out) const{    
    la::SubInit(r.pos_, q.pos_, force_out);
    if (box.length() == q.pos_.length()){
      AdjustVector_(force_out, box);
    }
    double dist = sqrt(la::Dot(*force_out, *force_out));  
    if (dist > sqrt(cutoff)){
      //      force_out->Init(3);
      force_out->SetZero();
      return  0;
    }
    double coef = 0, temp;
    for (int i  = 0; i < powers_.length(); i++){
      temp = -q.coefs_[i]*r.coefs_[i];
      coef = coef + signs_[i]*temp*powers_[i]*pow(dist, powers_[i]-2);
    }             
    la::Scale(coef / q.mass_, force_out);    
    return 1;
  }

  index_t n_terms() const{
    return powers_.length();
  }
  

};

#endif

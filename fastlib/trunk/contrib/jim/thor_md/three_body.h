#ifndef THREE_BODY_H
#define THREE_BODY_H

#include "fastlib/fastlib.h"

template<typename QNode, typename RNode, typename TPoint>
class ThreeBody{
 public:
  Vector box_size_;
  int periodic_;
  OT_DEF_BASIC(ThreeBody){
    OT_MY_OBJECT(periodic_);
    OT_MY_OBJECT(box_size_);
  }

 private:
 
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

  
  void AdjustVector_(Vector* vector_in) const{ 
    double L, x;
    for(int i = 0; i < 3; i++){
      L = box_size_[i];      
      x = (*vector_in)[i];
      if (unlikely(x  > L / 2)){
	(*vector_in)[i] -= L;
      } else { 
	if (unlikely(x  < -L / 2)){
	  (*vector_in)[i] += L;
	}
      }
    }
  }   
 

 public:
  void Init(const Vector& box_size_in){       
    if (box_size_in.length() == 3){
      box_size_.Init(3);
      for (int i = 0; i < 3; i++){
	box_size_[i] = box_size_in[i];
      }
      periodic_ = 1;
    } else {
      box_size_.Init(0);
      periodic_ = 0;
    }
  } 

  double ForceRange(const QNode& query, const RNode& ref1, const RNode& ref2)
    const{

    double range_i = 0;
    double Rij, Rjk, Rki, rij, rki, rjk, ri, rj, rk;
    double rnij = 0, rnjk = 0, rnki = 0, Rnij = 0, Rnjk = 0, Rnki = 0;
    Vector delta_ij, delta_jk, delta_ki;
    la::SubInit(query.stat().centroid_, ref1.stat().centroid_, &delta_ij);
    la::SubInit(ref1.stat().centroid_, ref2.stat().centroid_, &delta_jk);
    la::SubInit(ref2.stat().centroid_, query.stat().centroid_, &delta_ki); 
    if (periodic_){
      AdjustVector_(&delta_ij);
      AdjustVector_(&delta_jk);
      AdjustVector_(&delta_ki);
    }
    Rij = sqrt(la::Dot(delta_ij, delta_ij));
    Rjk = sqrt(la::Dot(delta_jk, delta_jk));
    Rki = sqrt(la::Dot(delta_ki, delta_ki));
    
    Vector bi, bj, bk;
    bi.Init(3);
    bj.Init(3);
    bk.Init(3);
    for (int d = 0; d < 3; d++){
      double size = 0;
      if (periodic_){
	size = box_size_[d];
      }
      bi[d] = query.bound().width(d, size) / 2;
      bj[d] = ref1.bound().width(d, size) / 2;
      bk[d] = ref2.bound().width(d, size) / 2;      
      rnij = rnij + bi[d] + bj[d];    
      rnki = rnki + bk[d] + bi[d];
      Rnij = Rnij + fabs(delta_ij[d]);     
      Rnki = Rnki + fabs(delta_ki[d]);
    }
    ri = la::Dot(bi, bi);
    rj = la::Dot(bj, bj);
    rk = la::Dot(bk, bk);
    rij = sqrt(ri+rj)/Rij;
    rjk = sqrt(rj+rk)/Rjk;
    rki = sqrt(rk+ri)/Rki;
    /*
      for (int d = 0; d < 10; d++){
      int a = abs((int)power3a_[d]);  
      int b = abs((int)power3b_[d]);  
      int c = abs((int)power3c_[d]);  
      double coef = i.stat().axilrod_[0]*j.stat().axilrod_[0]*
      k.stat().axilrod_[0]*signs3_[d];
      range_i += coef*GetPotentialTermPt_(Rjk, rjk, b)*
      (GetForceTerm_(Rij, rij, Rnij, rnij,a)*GetPotentialTerm_(Rki, rki,c)+
      GetForceTerm_(Rki, rki, Rnki, rnki,c)*GetPotentialTerm_(Rij, rij,a));
      }
    */    
    range_i = range_i / query.count(); 
    return range_i;
  }

  double PotentialRange(const QNode& query, const RNode& ref1,
			const RNode& ref2) const {
    double range_i;
    double Rij, Rjk, Rki, rij, rki, rjk;
    
    Vector delta_ij, delta_jk, delta_ki;
    la::SubInit(query.stat().centroid_, ref1.stat().centroid_, &delta_ij);
    la::SubInit(ref1.stat().centroid_, ref2.stat().centroid_, &delta_jk);
    la::SubInit(ref2.stat().centroid_, query.stat().centroid_, &delta_ki);
    if (periodic_){
      AdjustVector_(&delta_ij);
      AdjustVector_(&delta_jk);
      AdjustVector_(&delta_ki);
    }
    Rij = sqrt(la::Dot(delta_ij, delta_ij));
    Rjk = sqrt(la::Dot(delta_jk, delta_jk));
    Rki = sqrt(la::Dot(delta_ki, delta_ki));
    
    Vector bi, bj, bk;
    bi.Init(3);
    bj.Init(3);
    bk.Init(3);
    for (int d = 0; d < 3; d++){
      double size = 0;
      if (periodic_){
	size = box_size_[d];
      }
      bi[d] = query.bound().width(d, size) / 2;
      bj[d] = ref1.bound().width(d, size) / 2;
      bk[d] = ref2.bound().width(d, size) / 2; 
    }
    la::AddOverwrite(bi, bj, &delta_ij);
    la::AddOverwrite(bj, bk, &delta_jk);
    la::AddOverwrite(bk, bi, &delta_ki);
    rij = sqrt(la::Dot(delta_ij, delta_ij))/Rij;
    rjk = sqrt(la::Dot(delta_jk, delta_jk))/Rjk;
    rki = sqrt(la::Dot(delta_ki, delta_ki))/Rki;
    double dij, djk, dki;
    dij =   GetPotentialTerm_(Rij, rij, 3);
    dki =   GetPotentialTerm_(Rki, rki, 3);
    djk = GetPotentialTermPt_(Rjk, rjk, 3);
    
    double coef = query.stat().axilrod_[0]*ref1.stat().axilrod_[0]*
      ref2.stat().axilrod_[0]*6;
    range_i = coef*dij*dki*djk;
    
    range_i = range_i / query.count();
    
    return range_i;
    
  }
  
  void ForceVector(const QNode& q, const RNode& r1, const RNode& r2,
		   Vector* force_out) const{    
    Vector r_ij, r_jk, r_ki;
    la::SubInit( q.stat().centroid_, r1.stat().centroid_, &r_ij);
    la::SubInit(r1.stat().centroid_, r2.stat().centroid_, &r_jk);
    la::SubInit(r2.stat().centroid_,  q.stat().centroid_, &r_ki); 
    if (periodic_){
      AdjustVector_(&r_ij);
      AdjustVector_(&r_jk);
      AdjustVector_(&r_ki);
    }    
    double AA, BB, CC, AB, AC, BC, coef1, coef2;
    double cosines, denom;    
    // Extra Terms
    double denom2, coef1b, coef2b;
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
    denom = 3.0 * q.stat().axilrod_[0] * r1.stat().axilrod_[0] *
       r2.stat().axilrod_[0] / denom;          
    coef1 = denom*(2.0*AB*AC + BC*BC - 5.0*cosines/AA);
    coef2 = denom*(2.0*BC*AC + AB*AB - 5.0*cosines/CC);
    la::ScaleInit(-coef1, r_ij, force_out);
    la::AddExpert( coef2, r_ki, force_out);
     
    // Extra Term stuff    
    denom2 = 5.0 *  q.stat().axilrod_[1] * r1.stat().axilrod_[1] *
      r2.stat().axilrod_[1] / denom2;     
    coef1b = denom2*(BC*AA + BC*BC + 3.0*AC*AB - 14.0*cosines/AA);
    coef2b = denom2*(AB*CC + AB*AB + 3.0*AC*BC - 14.0*cosines/CC);   
    la::AddExpert(-coef1b, r_ij, force_out);
    la::AddExpert( coef2b, r_ki, force_out);

    la::Scale(1.0/q.stat().mass_, force_out);   
  }



  void ForceVector(const TPoint& q, const RNode& r1, const RNode& r2,
		   Vector* force_out) const{   
    Vector r_ij, r_jk, r_ki;
    la::SubInit(q.pos_, r1.stat().centroid_, &r_ij);
    la::SubInit(r1.stat().centroid_, r2.stat().centroid_, &r_jk);
    la::SubInit(r2.stat().centroid_, q.pos_, &r_ki);
    if (periodic_){
      AdjustVector_(&r_ij);
      AdjustVector_(&r_jk);
      AdjustVector_(&r_ki);
    }    
    double AA, BB, CC, AB, AC, BC, coef1, coef2;
    double cosines, denom;    
    // Extra Terms
    double denom2, coef1b, coef2b;
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
    denom = 3.0 * q.axilrod_[0] * r1.stat().axilrod_[0] *
      r2.stat().axilrod_[0] / denom;        
    coef1 = denom*(2.0*AB*AC + BC*BC - 5.0*cosines/AA);
    coef2 = denom*(2.0*BC*AC + AB*AB - 5.0*cosines/CC);	      
    la::ScaleInit(-coef1, r_ij, force_out);
    la::AddExpert( coef2, r_ki, force_out);
    
    // Extra Term stuff    
    denom2 = 5.0 *  q.axilrod_[1] * r1.stat().axilrod_[1] *
      r2.stat().axilrod_[1] / denom2;
    coef1b = denom2*(BC*AA + BC*BC + 3.0*AC*AB - 14.0*cosines/AA);
    coef2b = denom2*(AB*CC + AB*AB + 3.0*AC*BC - 14.0*cosines/CC);
    la::AddExpert(-coef1b, r_ij, force_out);
    la::AddExpert( coef2b, r_ki, force_out);   

    la::Scale(1.0/q.mass_, force_out);  
  }

  void ForceVector(const TPoint& q, const TPoint& r1, const TPoint& r2,
		   Vector* force_out) const{  
   
    Vector r_ij, r_jk, r_ki;  
    la::SubInit(q.pos_, r1.pos_, &r_ij);
    la::SubInit(r1.pos_, r2.pos_, &r_jk);
    la::SubInit(r2.pos_, q.pos_, &r_ki);
    if (periodic_){
      AdjustVector_(&r_ij);
      AdjustVector_(&r_jk);
      AdjustVector_(&r_ki);
    }   
    double AA, BB, CC, AB, AC, BC, coef1, coef2;
    double cosines, denom;
    
    // Extra Terms
    double denom2, coef1b, coef2b;   
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
    denom = 3.0 * q.axilrod_[0] * r1.axilrod_[0]* r2.axilrod_[0] / denom;   
    coef1 = denom*(2.0*AB*AC + BC*BC - 5.0*cosines/AA);
    coef2 = denom*(2.0*BC*AC + AB*AB - 5.0*cosines/CC);  
    la::ScaleInit(-coef1, r_ij, force_out);
    la::AddExpert( coef2, r_ki, force_out);      
      
    // Extra Term stuff      
    denom2 = 5.0 * q.axilrod_[1] * r1.axilrod_[1]* r2.axilrod_[1] / denom2;
    coef1b = denom2*(BC*AA + BC*BC + 3.0*AC*AB - 14.0*cosines/AA);
    coef2b = denom2*(AB*CC + AB*AB + 3.0*AC*BC - 14.0*cosines/CC);   
    la::AddExpert(-coef1b, r_ij, force_out);
    la::AddExpert( coef2b, r_ki, force_out); 

    la::Scale(1.0/q.mass_, force_out);   
  }


  int ForceVector(const TPoint& q, const TPoint& r1, const TPoint& r2, 
		  double cutoff, double cutoff2, Vector* force_out) const{   
    Vector r_ij, r_jk, r_ki;     
    r_ij.Init(3);
    r_jk.Init(3);
    r_ki.Init(3);
    la::SubOverwrite(q.pos_, r1.pos_, &r_ij);
    la::SubOverwrite(r1.pos_, r2.pos_, &r_jk);
    la::SubOverwrite(r2.pos_, q.pos_, &r_ki);
    if (periodic_){
      AdjustVector_(&r_ij);
      AdjustVector_(&r_jk);
      AdjustVector_(&r_ki);
    }   
    double AA, BB, CC, AB, AC, BC, coef1, coef2;
    double cosines, denom;
    
    // Extra Terms
    double denom2;   
    AA = la::Dot(r_ij, r_ij);
    CC = la::Dot(r_ki, r_ki);
    BB = la::Dot(r_jk, r_jk);    
    if (((AA > cutoff) || (BB > cutoff) || (CC > cutoff)) ||
	((AA > cutoff2) && (BB > cutoff2) && (CC > cutoff2))){
      force_out->Init(3);
      force_out->SetZero();    
      return 0;
    }     
    AC = 0.5*(BB - CC - AA);
    AB = 0.5*(CC - AA - BB);
    BC = 0.5*(AA - BB - CC);
    cosines = BC*AC*AB;
    denom = AA*BB*CC;
    denom2 = pow(denom, 3.5);      
    denom = pow(denom, 2.5);
    denom = 3.0 * q.axilrod_[0] * r1.axilrod_[0]* r2.axilrod_[0] / denom;   
    coef1 = denom*(2.0*AB*AC + BC*BC - 5.0*cosines/AA);
    coef2 = denom*(2.0*BC*AC + AB*AB - 5.0*cosines/CC);  
    la::ScaleInit(-coef1, r_ij, force_out);
    la::AddExpert( coef2, r_ki, force_out);      
      
    // Extra Term stuff      
    denom2 = 5.0 * q.axilrod_[1] * r1.axilrod_[1]* r2.axilrod_[1] / denom2;
    coef1 = denom2*(BC*AA + BC*BC + 3.0*AC*AB - 14.0*cosines/AA);
    coef2 = denom2*(AB*CC + AB*AB + 3.0*AC*BC - 14.0*cosines/CC);   
    la::AddExpert(-coef1, r_ij, force_out);
    la::AddExpert( coef2, r_ki, force_out);    

    la::Scale(1.0/q.mass_, force_out);  
    /*
    Vector foo;
    foo.Init(3);
    foo[0] = sqrt(AA);
    foo[1] = sqrt(BB);
    foo[2] = sqrt(CC);
    la::ScaleOverwrite(1.0, foo, force_out);
    */
    return 1;
  }

  

};

#endif

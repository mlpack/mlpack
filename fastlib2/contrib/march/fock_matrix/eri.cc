#include "eri.h"


double eri::F_0_(double z) {

  if (z == 0) {
    return 1.0;
  }
  else {
    return((1/sqrt(z)) * sqrt(math::PI) * 0.5 * erf(sqrt(z)));
  }
  
} // F_0_


double eri::SSSSIntegral(double alpha_A, const Vector& A_vec, double alpha_B, 
                         const Vector& B_vec, double alpha_C, 
                         const Vector& C_vec, double alpha_D, 
                         const Vector& D_vec) {

  double gamma_p = alpha_A + alpha_B;
  double gamma_q = alpha_C + alpha_D;

  double integral = 2*math::pow(math::Pi,2.5)/
                      (gamma_p*gamma_q*math::sqrt(gamma_p + gamma_q));

  double AB_dist = la::DistanceSqEuclidean(A_vec, B_vec);
  double CD_dist = la::DistanceSqEuclidean(C_vec, D_vec);

  
  Vector A_vec_scaled;
  la::ScaleInit(alpha_A, A_vec, &A_vec_scaled);
  Vector B_vec_scaled;
  la::ScaleInit(alpha_B, B_vec, &B_vec_scaled);
  Vector C_vec_scaled;
  la::ScaleInit(alpha_C, C_vec, &C_vec_scaled);
  Vector D_vec_scaled;
  la::ScaleInit(alpha_D, D_vec, &D_vec_scaled);

  
  Vector AB_vec;
  la::AddInit(A_vec_scaled, B_vec_scaled, &AB_vec);
  Vector CD_vec;
  la::AddInit(C_vec_scaled, D_vec_scaled, &CD_vec);

  double four_way_dist = la::DistanceSqEuclidean(AB_vec, CD_vec);
  
  integral = integral * 
      F_0_(four_way_dist * gamma_p * gamma_q/(gamma_p + gamma_q));
      
  double K1 = exp(-alpha_A * alpha_B * AB_dist/gamma_p);
  double K2 = exp(-alpha_C * alpha_D * CD_dist/gamma_q);
  
  integral = integral * K1 * K2;
  
  return integral;

}
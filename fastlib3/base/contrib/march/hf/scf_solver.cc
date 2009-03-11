#include "scf_solver.h"

double SCFSolver::ComputeOverlapIntegral_(double dist) {

  double integral = pow((math::PI/(2 * bandwidth_)), 1.5);
  
  integral = integral * exp(-0.5 * dist * bandwidth_);
  
  integral = integral * normalization_constant_squared_;
  
  return integral;

} //ComputeOverlapIntegral()


double SCFSolver::ComputeKineticIntegral_(double dist) {

  // I read Boys wrong, and the -1/2 was included in his computation
  double integral = pow((math::PI/(2 * bandwidth_)), 1.5);
  
  integral *= ((1.5 * bandwidth_) - (0.5 * bandwidth_ * bandwidth_ * dist));
  
  integral = integral * exp(-0.5 * dist * bandwidth_);
  
  integral = integral * normalization_constant_squared_;

  return integral;
  
} // ComputeKineticIntegral()


double SCFSolver::ComputeNuclearIntegral_(const Vector& nuclear_position, 
                                          index_t nuclear_index,
                                          const Vector& mu, const Vector& nu) {

  double integral = math::PI/bandwidth_;
  
  double basis_dist = la::DistanceSqEuclidean(mu, nu);
  
  integral *= exp(-0.5 * basis_dist * bandwidth_);

  Vector ave_center;
  la::AddInit(mu, nu, &ave_center);
  la::Scale(0.5, &ave_center);

  double nuclear_dist = la::DistanceSqEuclidean(nuclear_position, ave_center);

  integral *= integrals_.ErfLikeFunction(2 * bandwidth_ * nuclear_dist);
  
  integral *= -1 * nuclear_masses_[nuclear_index];
  
  integral = integral * normalization_constant_squared_;

  return integral;

} // ComputeNuclearIntegral()

void SCFSolver::ComputeOneElectronMatrices_() {
  
  for (index_t row_index = 0; row_index < number_of_basis_functions_; 
       row_index++) {
    
    for (index_t col_index = row_index; col_index < number_of_basis_functions_; 
         col_index++) {
      
      Vector row_vec;
      basis_centers_.MakeColumnVector(row_index, &row_vec);
      Vector col_vec;
      basis_centers_.MakeColumnVector(col_index, &col_vec);
      double dist = la::DistanceSqEuclidean(row_vec, col_vec);
      
      double kinetic_integral = ComputeKineticIntegral_(dist);
      
      double overlap_integral = ComputeOverlapIntegral_(dist);
      
      double nuclear_integral = 0.0;
      for (index_t nuclear_index = 0; nuclear_index < number_of_nuclei_; 
           nuclear_index++) {
        
        Vector nuclear_position;
        nuclear_centers_.MakeColumnVector(nuclear_index, &nuclear_position);
        
        nuclear_integral = nuclear_integral + 
          ComputeNuclearIntegral_(nuclear_position, nuclear_index, row_vec, 
                                  col_vec);
        
      } // nuclear_index
      
      kinetic_energy_integrals_.set(row_index, col_index, kinetic_integral);
      potential_energy_integrals_.set(row_index, col_index, nuclear_integral);
      overlap_matrix_.set(row_index, col_index, overlap_integral);
      
      if (likely(row_index != col_index)) {
        kinetic_energy_integrals_.set(col_index, row_index, kinetic_integral);
        potential_energy_integrals_.set(col_index, row_index, nuclear_integral);
        overlap_matrix_.set(col_index, row_index, overlap_integral);
      }
      
    } // column_index
    
  } // row_index
  
  la::AddInit(kinetic_energy_integrals_, potential_energy_integrals_, 
              &core_matrix_);
  
} // ComputeOneElectronMatrices_()



double SCFSolver::ComputeNuclearRepulsion_() {
  
  double nuclear_energy = 0.0;
  
  for (index_t a = 0; a < number_of_nuclei_; a++) {
    
    for (index_t b = a+1; b < number_of_nuclei_; b++) {
      
      Vector a_vec; 
      nuclear_centers_.MakeColumnVector(a, &a_vec);
      Vector b_vec;
      nuclear_centers_.MakeColumnVector(b, &b_vec);
      
      double dist_sq = la::DistanceSqEuclidean(a_vec, b_vec);
      double dist = sqrt(dist_sq);
      
      nuclear_energy += nuclear_masses_[a] * nuclear_masses_[b] / dist;
      
    } // b
    
  } // a
  
  return nuclear_energy;
  
} // ComputeNuclearRepulsion_()

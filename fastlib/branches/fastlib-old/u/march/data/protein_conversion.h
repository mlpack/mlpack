#ifndef PROTEIN_CONVERSION_H
#define PROTEIN_CONVERSION_H

#include <fastlib/fastlib.h>

/*
 Needs to:
 Take the structures one at a time
 Compute the features for that structure
 Write the features as a point in a dataset
 Write the dataset to a file
 
 the features:
 1-3 moments of center of mass
 4-6 moments of atom nearest center
 7-9 moments of atom farthest from center
 10-12 moments of atom farthest from farthest
 
 */

class Protein_Converter {
  
public:
  
  void Init(int num_structures, const char* output_file_name) {
    output_file = output_file_name;
    data.Init(num_features, num_structures);
    current_structure = 0;
  }
  
  void computeFeatures(Matrix structure) {
   
    //the matrix has x, y, z as its rows, each atom is a column
    
    index_t num_atoms = structure.n_cols();
    Vector center;
    center.Init(3);
    center.SetAll(0.0);
    
    for (index_t i = 0; i < num_atoms; i++) {
     
      Vector current;
      structure.MakeColumnVector(i, &current);
      la::AddTo(current, &center);
      
    }
    
    la::Scale((1.0/num_atoms), &center);
    
  /*  printf("center:\n");
    center.PrintDebug();
    */
    
    double min_dist = DBL_MAX;
    index_t min_index = -1;
    double max_dist = 0.0;
    index_t max_index = -1;    
    
    // Compute the atom nearest the center and the one farthest from it
    for (index_t i = 0; i < num_atoms; i++) {
     
      Vector current;
      structure.MakeColumnVector(i, &current);
      
      double current_dist;
            
      current_dist = la::DistanceSqEuclidean(current, center);
      if (current_dist < min_dist) {
        min_dist = current_dist;
        min_index = i;
      }
      if (current_dist > max_dist) {
        max_dist = current_dist;
        max_index = i;
      }
       
    }
    DEBUG_ASSERT(max_index > -1);
    DEBUG_ASSERT(min_index > -1);
    
    Vector center_atom;
    Vector far_atom_1;
    
    structure.MakeColumnVector(min_index, &center_atom);
    structure.MakeColumnVector(max_index, &far_atom_1);
    
    /*printf("center_atom:\n");
    center_atom.PrintDebug();
    printf("far_atom_1:\n");
    far_atom_1.PrintDebug();
    */
    
    max_dist = 0.0;
    max_index = -1;
    
    //compute the atom farthest from the previous outlier
    for (index_t i = 0; i < num_atoms; i++) {
     
      Vector current;
      structure.MakeColumnVector(i, &current);
      
      double current_dist;
      current_dist = la::DistanceSqEuclidean(current, far_atom_1);
      
      if (current_dist > max_dist) {
        max_dist = current_dist;
        max_index = i;
      }
      
    }
    DEBUG_ASSERT(max_index > -1);
    
    Vector far_atom_2;
    structure.MakeColumnVector(max_index, &far_atom_2);
    
    /*printf("far_atom_2:\n");
    far_atom_2.PrintDebug();
    */
    
    /* Now, compute the 4xnum_atoms matrix of distances from all the atoms to these four */
    
    Matrix distances;
    distances.Init(4, num_atoms);
    
    for (index_t i = 0; i < num_atoms; i++) {
     
      Vector current;
      structure.MakeColumnVector(i, &current);
      
      double center_dist, center_atom_dist, far_atom_dist_1, far_atom_dist_2;
      center_dist = sqrt(la::DistanceSqEuclidean(current, center));
      center_atom_dist = sqrt(la::DistanceSqEuclidean(current, center_atom));
      far_atom_dist_1 = sqrt(la::DistanceSqEuclidean(current, far_atom_1));
      far_atom_dist_2 = sqrt(la::DistanceSqEuclidean(current, far_atom_2));
      
      distances.set(0, i, center_dist);
      distances.set(1, i, center_atom_dist);
      distances.set(2, i, far_atom_dist_1);
      distances.set(3, i, far_atom_dist_2);
      
    }
    
    //distances.PrintDebug();
    
    /* Now, compute the new features, the first three sample moments of the distances matrix */
    
    double total1 = 0.0;
    double total2 = 0.0;
    double total3 = 0.0;
    double total4 = 0.0;
    
    for (index_t i = 0; i < num_atoms; i++) {
     
      total1 = total1 + distances.get(0, i);
      total2 = total2 + distances.get(1, i);
      total3 = total3 + distances.get(2, i);
      total4 = total4 + distances.get(3, i);
      
    }
    
    double sample_mean1 = total1/num_atoms;
    double sample_mean2 = total2/num_atoms;
    double sample_mean3 = total3/num_atoms;
    double sample_mean4 = total4/num_atoms;
    
    //printf("sample_mean1 = %f, 2 = %f, 3 = %f, 4 = %f\n", sample_mean1, sample_mean2, sample_mean3, sample_mean4);
    
    total1 = 0.0;
    total2 = 0.0;
    total3 = 0.0;
    total4 = 0.0;
    double skew_total1 = 0.0;
    double skew_total2 = 0.0;
    double skew_total3 = 0.0;
    double skew_total4 = 0.0;
    
    
    for (index_t i = 0; i < num_atoms; i++) {
     
      double temp;
      temp = distances.get(0, i) - sample_mean1;
      total1 = total1 + temp*temp;
      skew_total1 = skew_total1 + temp*temp*temp;
      
      temp = distances.get(1, i) - sample_mean2;
      total2 = total2 + temp*temp;
      skew_total2 = skew_total2 + temp*temp*temp;
      
      temp = distances.get(2, i) - sample_mean3;
      total3 = total3 + temp*temp;
      skew_total3 = skew_total3 + temp*temp*temp;
      
      temp = distances.get(3, i) - sample_mean4;
      total4 = total4 + temp*temp;
      skew_total4 = skew_total4 + temp*temp*temp;
      
    }
    
    DEBUG_ASSERT(num_atoms > 1);
    
    double sample_variance1 = total1/(num_atoms - 1);
    double sample_variance2 = total2/(num_atoms - 1);
    double sample_variance3 = total3/(num_atoms - 1);
    double sample_variance4 = total4/(num_atoms - 1);
    
    double sample_skew1 = sqrt((double)num_atoms) * skew_total1/(pow(sample_variance1, 1.5));
    double sample_skew2 = sqrt(num_atoms) * skew_total2/(pow(sample_variance2, 1.5));
    double sample_skew3 = sqrt(num_atoms) * skew_total3/(pow(sample_variance3, 1.5));
    double sample_skew4 = sqrt(num_atoms) * skew_total4/(pow(sample_variance4, 1.5));
    
    // printf("sample_skew1 = %f\n", sample_skew1);
    
    
    /* Now, all that is left is to write these into the output matrix, which is num_features x num_structures 
      * the matrix will have all four means, followed by variances, then skews */
    
    DEBUG_ASSERT(current_structure < data.n_cols());
    
    data.set(0, current_structure, sample_mean1);
    data.set(1, current_structure, sample_mean2);
    data.set(2, current_structure, sample_mean3);
    data.set(3, current_structure, sample_mean4);
    
    data.set(4, current_structure, sample_variance1);
    data.set(5, current_structure, sample_variance2);
    data.set(6, current_structure, sample_variance3);
    data.set(7, current_structure, sample_variance4);
    
    data.set(8, current_structure, sample_skew1);
    data.set(9, current_structure, sample_skew2);
    data.set(10, current_structure, sample_skew3);
    data.set(11, current_structure, sample_skew4);
    
    
    current_structure++;
    
    /*if (current_structure == data.n_cols()) {
      data.PrintDebug();
    }*/
    
  }
  
  void PrintData() {
   
    /* make sure to only call this after the data have been filled in */
    data::Save(output_file, data);
    
  }
  
  
  
private:
  const char* output_file;
  Matrix data;
  index_t current_structure;
  
  const static int num_features = 12;
  
};




#endif

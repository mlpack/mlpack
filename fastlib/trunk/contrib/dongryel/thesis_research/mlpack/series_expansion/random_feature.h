/** @file random_feature.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_RANDOM_FEATURE_H
#define MLPACK_SERIES_EXPANSION_RANDOM_FEATURE_H

namespace mlpack {
namespace series_expansion {
class RandomFeature {
  public:

    template<typename TableType, typename KernelType>
    static void Transform(
      const TableType &table_in,
      const KernelType &kernel_in,
      int num_random_fourier_features,
      TableType *table_out) {

      for(int i = 0; i < num_random_fourier_features; i++) {

        // Draw a random Fourier feature.
        arma::vec random_variate;
        kernel_in.DrawRandomVariate(
          table_in.n_attributes(), &random_variate);
      }
    }
};
}
}

#endif

/**
 * @file namespace_compat.hpp
 * @author Ryan Curtin
 *
 * This file is included for reverse compatibility with mlpack 3 and older code:
 * it introduces all the namespaces that were removed in mlpack 4.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_NAMESPACE_COMPAT_HPP
#define MLPACK_NAMESPACE_COMPAT_HPP

namespace mlpack {

// Namespaces from core/.
namespace bound { using namespace mlpack; }
namespace cv { using namespace mlpack; }
namespace distribution { using namespace mlpack; }
namespace hpt { using namespace mlpack; }
namespace kernel { using namespace mlpack; }
namespace math { using namespace mlpack; }
namespace metric { using namespace mlpack; }
namespace sfinae { using namespace mlpack; }
namespace tree { using namespace mlpack; }

// Namespaces from methods/.
namespace adaboost { using namespace mlpack; }
namespace amf { using namespace mlpack; }
namespace ann { using namespace mlpack; }
namespace cf { using namespace mlpack; }
namespace dbscan { using namespace mlpack; }
namespace det { using namespace mlpack; }
namespace emst { using namespace mlpack; }
namespace ensemble { using namespace mlpack; }
namespace fastmks { using namespace mlpack; }
namespace gmm { using namespace mlpack; }
namespace hmm { using namespace mlpack; }
namespace kde { using namespace mlpack; }
namespace kpca { using namespace mlpack; }
namespace kmeans { using namespace mlpack; }
namespace lmnn { using namespace mlpack; }
namespace lcc { using namespace mlpack; }
namespace matrix_completion { using namespace mlpack; }
namespace meanshift { using namespace mlpack; }
namespace naive_bayes { using namespace mlpack; }
namespace nca { using namespace mlpack; }
namespace neighbor { using namespace mlpack; }
namespace nn { using namespace mlpack; }
namespace pca { using namespace mlpack; }
namespace perceptron { using namespace mlpack; }
namespace radical { using namespace mlpack; }
namespace range { using namespace mlpack; }
namespace regression { using namespace mlpack; }
namespace sparse_coding { using namespace mlpack; }
namespace svd { using namespace mlpack; }
namespace svm { using namespace mlpack; }
namespace rl { using namespace mlpack; }

} // namespace mlpack

#endif

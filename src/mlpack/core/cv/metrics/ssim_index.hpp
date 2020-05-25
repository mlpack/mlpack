/**
 * @file ssim_index.hpp
 * @author Utkarsh Rai
 *
 * Definition of the SSIM (Structural Similarity) metric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_SSIM_HPP
#define MLPACK_CORE_CV_METRICS_SSIM_HPP

#include <mlpack/core.hpp>

namespace mlpack {
	namespace cv {

		class SSIMIndex {
			public:

				template<typename DataType>
					static double Evaluate(DataType const& image,
							DataType const& reference);

				static const bool NeedsMinimization = false;
		}; //class SSIMIndex
	} //namespace cv
} //namespace mlpack

#include "ssim_index_impl.hpp"

#endif

/**
 * @file methods/pdp/pdp.hpp
 * @author: Ankit Singh
 * 
 * Partial Dependene Plot (PDP) for mlpack models.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_PDP_PDP_HPP
#define MLPACK_METHODS_PDP_PDP_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
	
/**
* Partial Dependence Plot class.
* 
* The Partial Dependence Plot (short PDP or PD plot) shows the marginal 
* effect one or two features have on the predicted outcome of a machine 
* learning model (J. H. Friedman 2001).
* 
* DISADVANTAGES:
* It assumes features are independent of each other. (Use ALE Plot)
* The realistic maximum number of features in a partial dependence 
* function is two.
* Heterogeneous effects might be hidden because PD plots only show the 
* average marginal effects. (Use ICE Plot)
* 
* NOTE:
* Omitting the feature distribution can be misleading, because people might 
* overinterpret regions with almost no data. This problem is easily 
* solved by showing a rug (indicators for data points on the x-axis) or a histogram.
* 
* REFERENCES:
* Molnar, Christoph. "Interpretable machine learning. A Guide for Making 
* Black Box Models Explainable", 2019.
* https://christophm.github.io/interpretable-ml-book/.
*/
	/**
	* @tparam ModelType The type of the model.
	* @tparam Policy The policy class defining evaluation behaviour.
	*/
	template<typename ModelType,typename Policy>
	class PDP 
	{
	public:
		/**
		 * Constructor.
		 *
		 * @param model The trained model.
		 * @param data The dataset used for computing PDP.
		 * @param featureIndex The index of the feature for which to compute PDP.
		 * @param numPoints The number of points to sample along the feature range.
		 */

		PDP(const ModelType& model,
			const arma::mat& data,
			const size_t featureIndex,
			const size_t numPoints = 100);

		/**
	  * Compute partial dependence values.
	  *
	  * @return A tuple containing feature values and corresponding PDP values.
	  */
		std::tuple<arma::vec, arma::vec> Compute();

		/**
		 * Enable or disable ICE plots.
		 */
		void EnableICE(bool enable);

		/**
		 * Set the histogram flag.
		 */
		void EnableHistogram(bool enable);

	private:
		//! The trained model.
		const ModelType& model;
		//! Dataset.
		const arma::mat& data;
		//! Feature index for PDP computation.
		size_t featureIndex;
		//! Number of points to sample.
		size_t numPoints;
		//! ICE plot flag.
		bool iceEnabled;
		//! Histogram flag.
		bool histogramEnabled;

	};

} // namespace mlpack

#include "pdp_impl.hpp"

#endif
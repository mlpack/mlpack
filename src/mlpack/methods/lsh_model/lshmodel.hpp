/**
 * @file lshmodel.hpp
 * @author Yannis Mentekidis
 *
 * Defines the LSHModel class, which models the Locality Sensitive Hashing
 * algorithm. The model identifies parameter sets that produce satisfactory
 * results while keeping execution time low.
 * 
 * The model was proposed by Dong et al in the following paper.
 *
 * @code
 * @article{Dong2008LSHModel,
 *  author = {Dong, Wei and Wang, Zhe and Josephson, William and Charikar, 
 *      Moses and Li, Kai},
 *  title = {{Modeling LSH for performance tuning}},
 *  journal = {Proceeding of the 17th ACM conference on Information and 
 *      knowledge mining - CIKM '08},
 *  pages = {669},
 *  url = {http://portal.acm.org/citation.cfm?doid=1458082.1458172},
 *  year = {2008}
 * }
 * @endcode
 *
 * We use a different method to fit Gamma Distributions to pairwise distances.
 * Instead of the MLE method proposed in the paper above, we use the mlpack
 * class GammaDistribution, which implements fitting according to Thomas Minka's
 * work.
 *
 * @code
 * @techreport{minka2002estimating,
 *   title={Estimating a {G}amma distribution},
 *   author={Minka, Thomas P.},
 *   institution={Microsoft Research},
 *   address={Cambridge, U.K.},
 *   year={2002}
 * }
 * @endcode
 */


namespace mlpack {
namespace neighbor {

class LSHModel
{
 public:


 private:


}; // class LSHModel.

} // namespace neighbor.
} // namespace mlpack.

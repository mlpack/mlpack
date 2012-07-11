/**
 * @file nmf.cpp
 * @author Mohan Rajendran
 *
 * Implementation of NMF class to perform Non-Negative Matrix Factorization
 * on the given matrix.
 */

namespace mlpack {
namespace nmf {

/**
 * Construct the NMF object.
 */
template<typename InitializeRule,
         typename WUpdateRule,
         typename HUpdateRule>
NMF<InitializeRule,
    WUpdateRule,
    HUpdateRule>::
NMF(const size_t maxIterations,
      const double maxResidue,
      const InitializeRule Initialize,
      const WUpdateRule WUpdate,
      const HUpdateRule HUpdate) :
    maxIterations(maxIterations),
    maxResidue(maxResidue),
    Initialize(Initialize),
    WUpdate(WUpdate),
    HUpdate(HUpdate)
{
  if (maxResidue < 0.0)
  {
    Log::Warn << "NMF::NMF(): maxResidue must be a positive value ("
        << maxResidue << " given). Setting to the default value of "
        << "1e-10.\n";
    this->maxResidue = 1e-10;
  } 

  math::RandomSeed((size_t) std::time(NULL));    
}

/**
 * Apply the Non-Negative Matrix Factorization on the provided matrix.
 *
 * @param V Input matrix to be factorized
 * @param W Basis matrix to be output
 * @param H Encoding matrix to output
 * @param r Rank r of the factorization
 */
template<typename InitializeRule,
         typename WUpdateRule,
         typename HUpdateRule>
void NMF<InitializeRule,
    WUpdateRule,
    HUpdateRule>::
Apply(const arma::mat& V, arma::mat& W, arma::mat& H, size_t& r) const
{
  size_t n = V.n_rows;
  size_t m = V.n_cols;

  // Intialize W and H
  Initialize.Init(V,W,H,r);

  //Log::Debug << "Initialized W and H." << std::endl;

  size_t iteration = 0;
  size_t nm = n*m;
  double residue = maxResidue;
  double normOld,norm;
  arma::mat WH;    
  
  while (residue >= maxResidue  && iteration != maxIterations)
  {
    // Update step.
    // Update the value of W and H based on the Update Rules provided
    WUpdate.Update(V,W,H);
    HUpdate.Update(V,W,H);

    // Calculate norm of WH after each iteration
    WH = W*H;
    norm = sqrt(accu(WH%WH)/nm);
    
    if(iteration!=0)
    {
      residue = fabs(normOld-norm);
      if(normOld > 1.0)
      {
        residue /= normOld;
      }
    }

    normOld = norm;

    /*
      WH = W*H;
      diff = WHold-WH;
    diff = diff%diff;
    residue = accu(diff)/(double)(n*m);
    WHold = WH;*/

    //Log::Debug << "Iteration: " << iteration << " Residue: " 
    //      << sqrt(residue) << std::endl;

    iteration++;
      
  }

  //Log::Debug << "Iterations: " << iteration << std::endl;
}

}; // namespace nmf
}; // namespace mlpack

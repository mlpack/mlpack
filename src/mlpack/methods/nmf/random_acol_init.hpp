/**
 * @file randomacolinit.hpp
 * @author Mohan Rajendran
 *
 * Intialization rule for the Non-negative Matrix Factorization. This simple
 * initialization is performed by the rendom Acol initialization introduced in
 * the paper 'Algorithms, Initializations and Convergence' by Langville et al.
 * This method sets each of the column of W by averaging p randomly chosen
 * columns of A.
 */

#ifndef __MLPACK_METHODS_NMF_RANDOMACOLINIT_HPP
#define __MLPACK_METHODS_NMF_RANDOMACOLINIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace nmf {

class RandomAcolInitialization
{
 public:
  // Empty constructor required for the InitializeRule template
  RandomAcolInitialization()
  { }

  inline static void Initialize(const arma::mat& V,
                                const size_t r,
                                arma::mat& W,
                                arma::mat& H)
  {
    // Simple implementation. This can be left here.

    size_t n = V.n_rows;
    size_t m = V.n_cols;

    size_t p = 5;    

    if(p > m)
    {
      Log::Info << "No. of random columns is more than the number of columns "
          << "available in the V matrix. Setting the no. of random columns "
          << "to " << m << "." << std::endl;
      p = m;
    }

    W.reset();
    
    // Initialize W matrix
    arma::vec temp;
    for(size_t col=0;col<r;col++)
    {
      temp.zeros(n);
      for(size_t randcol=0;randcol<p;randcol++)
      {
        size_t rnd = math::RandInt(0,m);
        temp += V.col(rnd);
      }
      W.insert_cols(col,temp/p);
    }    
  
    // Intialize H to random values
    H.randu(r,m);
  }
  
}; // Class RandomAcolInitialization

}; // namespace nmf
}; // namespace mlpack

#endif

/*namespace mlpack {
namespace nmf {

class RandomAcolInitialization
{
 private:
  size_t p;
 public:
  // Constructor required for the InitializeRule template
  RandomAcolInitialization()
  { }

  inline void Init(const arma::mat& V,
                     arma::mat& W,
                     arma::mat& H,
                     const size_t& r)
  {
    // Simple inplementation. This can be left here.
    size_t n = V.n_rows;
    size_t m = V.n_cols;
    p = 5;
    if(p > m)
    {
      Log::Info << "No. of random columns is more than the number of columns "
          << "available in the V matrix. Setting the no. of random columns "
          << "to " << m << "." << std::endl;
      p = m;
    }
    
    // Initialize W matrix
    W.zeros(n,r);
    arma::colvec temp;
    for(size_t col=0;col<r;col++)
    {
      temp.zeros();
      for(size_t randcol=0;randcol<p;randcol++)
      {
        size_t rnd = math::RandInt(0,m);
        temp += V.col(rnd);
      }
      W.insert_cols(col,temp/p);
    }
  
    // Intialize H to random values
    H.randu(r,m);
  }
  
}; // Class RandomInitialization

}; // namespace nmf
}; // namespace mlpack

#endif*/

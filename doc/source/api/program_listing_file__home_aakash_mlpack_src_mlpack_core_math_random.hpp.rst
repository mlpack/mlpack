
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_math_random.hpp:

Program Listing for File random.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_math_random.hpp>` (``/home/aakash/mlpack/src/mlpack/core/math/random.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_MATH_RANDOM_HPP
   #define MLPACK_CORE_MATH_RANDOM_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/mlpack_export.hpp>
   #include <random>
   
   namespace mlpack {
   namespace math  {
   
   // Global random object.
   extern MLPACK_EXPORT std::mt19937 randGen;
   // Global uniform distribution.
   extern MLPACK_EXPORT std::uniform_real_distribution<> randUniformDist;
   // Global normal distribution.
   extern MLPACK_EXPORT std::normal_distribution<> randNormalDist;
   
   inline void RandomSeed(const size_t seed)
   {
     #if (!defined(BINDING_TYPE) || BINDING_TYPE != BINDING_TYPE_TEST)
       randGen.seed((uint32_t) seed);
       #if (BINDING_TYPE == BINDING_TYPE_R)
         // To suppress Found 'srand', possibly from 'srand' (C).
         (void) seed;
       #else
         srand((unsigned int) seed);
       #endif
       arma::arma_rng::set_seed(seed);
     #else
       (void) seed;
     #endif
   }
   
   #if (BINDING_TYPE == BINDING_TYPE_TEST)
   inline void FixedRandomSeed()
   {
     const static size_t seed = rand();
     randGen.seed((uint32_t) seed);
     srand((unsigned int) seed);
     arma::arma_rng::set_seed(seed);
   }
   
   inline void CustomRandomSeed(const size_t seed)
   {
     randGen.seed((uint32_t) seed);
     srand((unsigned int) seed);
     arma::arma_rng::set_seed(seed);
   }
   #endif
   
   inline double Random()
   {
     return randUniformDist(randGen);
   }
   
   inline double Random(const double lo, const double hi)
   {
     return lo + (hi - lo) * randUniformDist(randGen);
   }
   
   inline double RandBernoulli(const double input)
   {
     if (Random() < input)
       return 1;
     else
       return 0;
   }
   
   inline int RandInt(const int hiExclusive)
   {
     return (int) std::floor((double) hiExclusive * randUniformDist(randGen));
   }
   
   inline int RandInt(const int lo, const int hiExclusive)
   {
     return lo + (int) std::floor((double) (hiExclusive - lo)
                                  * randUniformDist(randGen));
   }
   
   inline double RandNormal()
   {
     return randNormalDist(randGen);
   }
   
   inline double RandNormal(const double mean, const double variance)
   {
     return variance * randNormalDist(randGen) + mean;
   }
   
   inline void ObtainDistinctSamples(const size_t loInclusive,
                                     const size_t hiExclusive,
                                     const size_t maxNumSamples,
                                     arma::uvec& distinctSamples)
   {
     const size_t samplesRangeSize = hiExclusive - loInclusive;
   
     if (samplesRangeSize > maxNumSamples)
     {
       arma::Col<size_t> samples;
   
       samples.zeros(samplesRangeSize);
   
       for (size_t i = 0; i < maxNumSamples; ++i)
         samples [ (size_t) math::RandInt(samplesRangeSize) ]++;
   
       distinctSamples = arma::find(samples > 0);
   
       if (loInclusive > 0)
         distinctSamples += loInclusive;
     }
     else
     {
       distinctSamples.set_size(samplesRangeSize);
       for (size_t i = 0; i < samplesRangeSize; ++i)
         distinctSamples[i] = loInclusive + i;
     }
   }
   
   } // namespace math
   } // namespace mlpack
   
   #endif // MLPACK_CORE_MATH_MATH_LIB_HPP

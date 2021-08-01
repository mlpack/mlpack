
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_amf_termination_policies_simple_tolerance_termination.hpp:

Program Listing for File simple_tolerance_termination.hpp
=========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_amf_termination_policies_simple_tolerance_termination.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/amf/termination_policies/simple_tolerance_termination.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef _MLPACK_METHODS_AMF_SIMPLE_TOLERANCE_TERMINATION_HPP_INCLUDED
   #define _MLPACK_METHODS_AMF_SIMPLE_TOLERANCE_TERMINATION_HPP_INCLUDED
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace amf {
   
   template <class MatType>
   class SimpleToleranceTermination
   {
    public:
     SimpleToleranceTermination(const double tolerance = 1e-5,
                                const size_t maxIterations = 10000,
                                const size_t reverseStepTolerance = 3) :
         tolerance(tolerance),
         maxIterations(maxIterations),
         V(nullptr),
         iteration(1),
         residueOld(DBL_MAX),
         residue(DBL_MIN),
         reverseStepTolerance(reverseStepTolerance),
         reverseStepCount(0),
         isCopy(false),
         c_indexOld(0),
         c_index(0)
     { }
   
     void Initialize(const MatType& V)
     {
       residueOld = DBL_MAX;
       iteration = 1;
       residue = DBL_MIN;
       reverseStepCount = 0;
       isCopy = false;
   
       this->V = &V;
   
       c_index = 0;
       c_indexOld = 0;
     }
   
     bool IsConverged(arma::mat& W, arma::mat& H)
     {
       arma::mat WH;
   
       WH = W * H;
   
       // Compute residue.
       residueOld = residue;
       size_t n = V->n_rows;
       size_t m = V->n_cols;
       double sum = 0;
       size_t count = 0;
       for (size_t i = 0; i < n; ++i)
       {
           for (size_t j = 0; j < m; ++j)
           {
               double temp = 0;
               if ((temp = (*V)(i, j)) != 0)
               {
                   temp = (temp - WH(i, j));
                   temp = temp * temp;
                   sum += temp;
                   count++;
               }
           }
       }
   
       residue = sum;
       if (count > 0)
         residue /= count;
       residue = sqrt(residue);
   
       // Increment iteration count.
       iteration++;
       Log::Info << "Iteration " << iteration << "; residue "
           << ((residueOld - residue) / residueOld) << ".\n";
   
       // If residue tolerance is not satisfied.
       if ((residueOld - residue) / residueOld < tolerance && iteration > 4)
       {
         // Check if this is a first of successive drops.
         if (reverseStepCount == 0 && isCopy == false)
         {
           // Store a copy of W and H matrix.
           isCopy = true;
           this->W = W;
           this->H = H;
           // Store residue values.
           c_index = residue;
           c_indexOld = residueOld;
         }
         // Increase successive drop count.
         reverseStepCount++;
       }
       // If tolerance is satisfied.
       else
       {
         // Initialize successive drop count.
         reverseStepCount = 0;
         // If residue is droped below minimum scrap stored values.
         if (residue <= c_indexOld && isCopy == true)
         {
           isCopy = false;
         }
       }
   
       // Check if termination criterion is met.
       if (reverseStepCount == reverseStepTolerance || iteration > maxIterations)
       {
         // If stored values are present replace them with current value as they
         // represent the minimum residue point.
         if (isCopy)
         {
           W = this->W;
           H = this->H;
           residue = c_index;
         }
         return true;
       }
   
       return false;
     }
   
     const double& Index() const { return residue; }
   
     const size_t& Iteration() const { return iteration; }
   
     const size_t& MaxIterations() const { return maxIterations; }
     size_t& MaxIterations() { return maxIterations; }
   
     const double& Tolerance() const { return tolerance; }
     double& Tolerance() { return tolerance; }
   
    private:
     double tolerance;
     size_t maxIterations;
   
     const MatType* V;
   
     size_t iteration;
   
     double residueOld;
     double residue;
   
     size_t reverseStepTolerance;
     size_t reverseStepCount;
   
     bool isCopy;
   
     arma::mat W;
     arma::mat H;
     double c_indexOld;
     double c_index;
   }; // class SimpleToleranceTermination
   
   } // namespace amf
   } // namespace mlpack
   
   #endif // _MLPACK_METHODS_AMF_SIMPLE_TOLERANCE_TERMINATION_HPP_INCLUDED
   

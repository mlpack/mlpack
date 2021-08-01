
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_amf_termination_policies_validation_rmse_termination.hpp:

Program Listing for File validation_rmse_termination.hpp
========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_amf_termination_policies_validation_rmse_termination.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/amf/termination_policies/validation_rmse_termination.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef _MLPACK_METHODS_AMF_VALIDATIONRMSETERMINATION_HPP_INCLUDED
   #define _MLPACK_METHODS_AMF_VALIDATIONRMSETERMINATION_HPP_INCLUDED
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack
   {
   namespace amf
   {
   
   template <class MatType>
   class ValidationRMSETermination
   {
    public:
     ValidationRMSETermination(MatType& V,
                               size_t num_test_points,
                               double tolerance = 1e-5,
                               size_t maxIterations = 10000,
                               size_t reverseStepTolerance = 3)
           : tolerance(tolerance),
             maxIterations(maxIterations),
             num_test_points(num_test_points),
             reverseStepTolerance(reverseStepTolerance)
     {
       size_t n = V.n_rows;
       size_t m = V.n_cols;
   
       // initialize validation set matrix
       test_points.zeros(num_test_points, 3);
   
       // fill validation set matrix with random chosen entries
       for (size_t i = 0; i < num_test_points; ++i)
       {
         double t_val;
         size_t t_row;
         size_t t_col;
   
         // pick a random non-zero entry
         do
         {
           t_row = rand() % n;
           t_col = rand() % m;
         } while ((t_val = V(t_row, t_col)) == 0);
   
         // add the entry to the validation set
         test_points(i, 0) = t_row;
         test_points(i, 1) = t_col;
         test_points(i, 2) = t_val;
   
         // nullify the added entry from data matrix (training set)
         V(t_row, t_col) = 0;
       }
     }
   
     void Initialize(const MatType& /* V */)
     {
       iteration = 1;
   
       rmse = DBL_MAX;
       rmseOld = DBL_MAX;
   
       c_index = 0;
       c_indexOld = 0;
   
       reverseStepCount = 0;
       isCopy = false;
     }
   
     bool IsConverged(arma::mat& W, arma::mat& H)
     {
       arma::mat WH;
   
       WH = W * H;
   
       // compute validation RMSE
       if (iteration != 0)
       {
         rmseOld = rmse;
         rmse = 0;
         for (size_t i = 0; i < num_test_points; ++i)
         {
           size_t t_row = test_points(i, 0);
           size_t t_col = test_points(i, 1);
           double t_val = test_points(i, 2);
           double temp = (t_val - WH(t_row, t_col));
           temp *= temp;
           rmse += temp;
         }
         rmse /= num_test_points;
         rmse = sqrt(rmse);
       }
   
       // increment iteration count
       iteration++;
   
       // if RMSE tolerance is not satisfied
       if ((rmseOld - rmse) / rmseOld < tolerance && iteration > 4)
       {
         // check if this is a first of successive drops
         if (reverseStepCount == 0 && isCopy == false)
         {
           // store a copy of W and H matrix
           isCopy = true;
           this->W = W;
           this->H = H;
           // store residue values
           c_indexOld = rmseOld;
           c_index = rmse;
         }
         // increase successive drop count
         reverseStepCount++;
       }
       // if tolerance is satisfied
       else
       {
         // initialize successive drop count
         reverseStepCount = 0;
         // if residue is droped below minimum scrap stored values
         if (rmse <= c_indexOld && isCopy == true)
         {
           isCopy = false;
         }
       }
   
       // check if termination criterion is met
       if (reverseStepCount == reverseStepTolerance || iteration > maxIterations)
       {
         // if stored values are present replace them with current value as they
         // represent the minimum residue point
         if (isCopy)
         {
           W = this->W;
           H = this->H;
           rmse = c_index;
         }
         return true;
       }
       else return false;
     }
   
     const double& Index() const { return rmse; }
   
     const size_t& Iteration() const { return iteration; }
   
     const size_t& NumTestPoints() const { return num_test_points; }
   
     const size_t& MaxIterations() const { return maxIterations; }
     size_t& MaxIterations() { return maxIterations; }
   
     const double& Tolerance() const { return tolerance; }
     double& Tolerance() { return tolerance; }
   
    private:
     double tolerance;
     size_t maxIterations;
     size_t num_test_points;
   
     size_t iteration;
   
     arma::mat test_points;
   
     double rmseOld;
     double rmse;
   
     size_t reverseStepTolerance;
     size_t reverseStepCount;
   
     bool isCopy;
   
     arma::mat W;
     arma::mat H;
     double c_indexOld;
     double c_index;
   }; // class ValidationRMSETermination
   
   } // namespace amf
   } // namespace mlpack
   
   
   #endif // _MLPACK_METHODS_AMF_VALIDATIONRMSETERMINATION_HPP_INCLUDED

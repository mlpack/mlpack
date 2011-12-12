
#include <mlpack/core.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>

#include "kpca.hpp"

using namespace mlpack;
using namespace std;
using namespace arma;
using namespace kpca;



int main(int argc, char** argv)
{

  /*mat data("1 0 2 3 9;"
            "5 2 8 4 8;"
            "6 7 3 1 8");*/
  mat data("1 2 3;"
            "4 5 6;"
            "7 8 9");

  data.print("DATA : ");

   // Now run PCA to reduce the dimensionality.
   kpca::KPCA<kernel::LinearKernel> p;
   //p.CenterData();
   p.Apply(data, 2); // Reduce to 2 dimensions.

   data.print("RESULT : ");
   // Compare with correct results.
   mat correct("-1.53781086 -3.51358020 -0.16139887 -1.87706634  7.08985628;"
               " 1.29937798  3.45762685 -2.69910005 -3.15620704  1.09830225");
   correct.print("CORRECT");

   // If the eigenvectors are pointed opposite directions, they will cancel
 // each other out in this summation.
   for(size_t i = 0; i < data.n_rows; i++)
   {
     if (fabs(correct(i, 1) + data(i,1)) < 0.001 /* arbitrary */)
     {
          // Flip Armadillo coefficients for this column.
          data.row(i) *= -1;
     }
   }
}

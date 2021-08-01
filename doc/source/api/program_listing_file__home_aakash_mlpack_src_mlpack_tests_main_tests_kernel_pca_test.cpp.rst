
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_main_tests_kernel_pca_test.cpp:

Program Listing for File kernel_pca_test.cpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_main_tests_kernel_pca_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/main_tests/kernel_pca_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   
   #define BINDING_TYPE BINDING_TYPE_TEST
   static const std::string testName = "KernelPrincipalComponentsAnalysis";
   
   #include <mlpack/core/util/mlpack_main.hpp>
   #include "test_helper.hpp"
   #include <mlpack/methods/kernel_pca/kernel_pca_main.cpp>
   
   #include "../catch.hpp"
   #include "../test_catch_tools.hpp"
   
   using namespace mlpack;
   
   struct KernelPCATestFixture
   {
    public:
     KernelPCATestFixture()
     {
       // Cache in the options for this program.
       IO::RestoreSettings(testName);
     }
   
     ~KernelPCATestFixture()
     {
       // Clear the settings.
       bindings::tests::CleanMemory();
       IO::ClearSettings();
     }
   };
   
   static void ResetSettings()
   {
     bindings::tests::CleanMemory();
     IO::ClearSettings();
     IO::RestoreSettings(testName);
   }
   
   TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCADimensionTest",
                    "[KernelPCAMainTest][BindingTests]")
   {
     std::string kernels[] = {
         "linear", "gaussian", "polynomial",
         "hyptan", "laplacian", "epanechnikov", "cosine"
     };
   
     for (std::string& kernel : kernels)
     {
       ResetSettings();
       arma::mat x = arma::randu<arma::mat>(5, 5);
       // Random input, new dimensionality of 3.
       SetInputParam("input", std::move(x));
       SetInputParam("new_dimensionality", (int) 3);
       SetInputParam("kernel", kernel);
       mlpackMain();
   
       // Now check that the output has 3 dimensions.
       REQUIRE(IO::GetParam<arma::mat>("output").n_rows == 3);
       REQUIRE(IO::GetParam<arma::mat>("output").n_cols == 5);
     }
   }
   
   TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCANoKernelTest",
                    "[KernelPCAMainTest][BindingTests]")
   {
     arma::mat x = arma::randu<arma::mat>(5, 5);
   
     SetInputParam("input", std::move(x));
     SetInputParam("new_dimensionality", (int) 3);
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCAInvalidKernelTest",
                    "[KernelPCAMainTest][BindingTests]")
   {
     arma::mat x = arma::randu<arma::mat>(5, 5);
   
     SetInputParam("input", std::move(x));
     SetInputParam("new_dimensionality", (int) 3);
     SetInputParam("kernel", (std::string) "badName");
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCA0DimensionalityTest",
                    "[KernelPCAMainTest][BindingTests]")
   {
     arma::mat x = arma::randu<arma::mat>(5, 5);
   
     SetInputParam("input", std::move(x));
     SetInputParam("new_dimensionality", (int) 0);
     SetInputParam("kernel", (std::string) "gaussian");
     mlpackMain();
   
     // Now check that the output has same dimensions as input.
     REQUIRE(IO::GetParam<arma::mat>("output").n_rows == 5);
     REQUIRE(IO::GetParam<arma::mat>("output").n_cols == 5);
   }
   
   TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCACenterTest",
                    "[KernelPCAMainTest][BindingTests]")
   {
     arma::mat x = arma::randu<arma::mat>(5, 5);
   
     // Get output without centering the dataset.
     SetInputParam("input", x);
     SetInputParam("new_dimensionality", (int) 3);
     SetInputParam("kernel", (std::string) "linear");
     mlpackMain();
     arma::mat output1 = IO::GetParam<arma::mat>("output");
   
     // Get output after centering the dataset.
     SetInputParam("input", std::move(x));
     SetInputParam("center", true);
     mlpackMain();
     arma::mat output2 = IO::GetParam<arma::mat>("output");
   
     // The resulting matrices should be different.
     REQUIRE(arma::any(arma::vectorise(output1 != output2)));
   }
   
   TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCATooHighNewDimensionalityTest",
                    "[KernelPCAMainTest][BindingTests]")
   {
     arma::mat x = arma::randu<arma::mat>(5, 5);
   
     SetInputParam("input", std::move(x));
     SetInputParam("new_dimensionality", (int) 7); // Invalid.
     SetInputParam("kernel", (std::string) "linear");
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCANoInputTest",
                    "[KernelPCAMainTest][BindingTests]")
   {
     arma::mat x = arma::randu<arma::mat>(5, 5);
   
     SetInputParam("new_dimensionality", (int) 2);
     SetInputParam("kernel", (std::string) "linear");
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCABadSamplingTest",
                    "[KernelPCAMainTest][BindingTests]")
   {
     arma::mat x = arma::randu<arma::mat>(5, 5);
   
     SetInputParam("input", std::move(x));
     SetInputParam("new_dimensionality", (int) 3);
     SetInputParam("kernel", (std::string) "linear");
     SetInputParam("nystroem_method", true);
     SetInputParam("sampling", (std::string) "badName");
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCABandWidthTest",
                    "[KernelPCAMainTest][BindingTests]")
   {
     std::string kernels[] = {
         "gaussian", "epanechnikov", "laplacian"
     };
   
     for (std::string& kernel : kernels)
     {
       ResetSettings();
       arma::mat x = arma::randu<arma::mat>(5, 5);
   
       // Get output using bandwidth 1.
       SetInputParam("input", x);
       SetInputParam("new_dimensionality", (int) 3);
       SetInputParam("kernel", kernel);
       SetInputParam("bandwidth", (double) 1);
   
       mlpackMain();
       arma::mat output1 = IO::GetParam<arma::mat>("output");
   
       // Get output using bandwidth 2.
       SetInputParam("input", std::move(x));
       SetInputParam("bandwidth", (double) 2);
   
       mlpackMain();
       arma::mat output2 = IO::GetParam<arma::mat>("output");
   
       // The resulting matrices should be different.
       REQUIRE(arma::any(arma::vectorise(output1 != output2)));
     }
   }
   
   TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCAOffsetTest",
                    "[KernelPCAMainTest][BindingTests]")
   {
     std::string kernels[] = {
         "polynomial", "hyptan"
     };
   
     for (std::string& kernel : kernels)
     {
       ResetSettings();
       arma::mat x = arma::randu<arma::mat>(5, 100);
   
       SetInputParam("input", x);
       SetInputParam("new_dimensionality", (int) 3);
       SetInputParam("kernel", kernel);
       SetInputParam("offset", (double) 0.01);
   
       mlpackMain();
       arma::mat output1 = IO::GetParam<arma::mat>("output");
   
       SetInputParam("input", std::move(x));
       SetInputParam("offset", (double) 0.1);
   
       mlpackMain();
       arma::mat output2 = IO::GetParam<arma::mat>("output");
   
       // The resulting matrices should be different.
       REQUIRE(arma::any(arma::vectorise(output1 != output2)));
     }
   }
   
   TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCADegreeTest",
                    "[KernelPCAMainTest][BindingTests]")
   {
     arma::mat x = arma::randu<arma::mat>(5, 5);
   
     SetInputParam("input", x);
     SetInputParam("new_dimensionality", (int) 3);
     SetInputParam("kernel", (std::string) "polynomial");
     SetInputParam("degree", (double) 2);
   
     mlpackMain();
     arma::mat output1 = IO::GetParam<arma::mat>("output");
   
     SetInputParam("input", std::move(x));
     SetInputParam("degree", (double) 3);
   
     mlpackMain();
     arma::mat output2 = IO::GetParam<arma::mat>("output");
   
     // The resulting matrices should be different.
     REQUIRE(arma::any(arma::vectorise(output1 != output2)));
   }
   
   TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCAKernelScaleTest",
                    "[KernelPCAMainTest][BindingTests]")
   {
     arma::mat x = arma::randu<arma::mat>(5, 5);
   
     SetInputParam("input", x);
     SetInputParam("new_dimensionality", (int) 3);
     SetInputParam("kernel", (std::string) "hyptan");
     SetInputParam("kernel_scale", (double) 2);
   
     mlpackMain();
     arma::mat output1 = IO::GetParam<arma::mat>("output");
   
     SetInputParam("input", std::move(x));
     SetInputParam("kernel_scale", (double) 3);
   
     mlpackMain();
     arma::mat output2 = IO::GetParam<arma::mat>("output");
   
     // The resulting matrices should be different.
     REQUIRE(arma::any(arma::vectorise(output1 != output2)));
   }
   
   TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCASamplingSchemeTest",
                    "[KernelPCAMainTest][BindingTests]")
   {
     ResetSettings();
   
     arma::mat x = arma::randu<arma::mat>(5, 500);
   
     SetInputParam("input", x);
     SetInputParam("new_dimensionality", (int) 3);
     SetInputParam("kernel", (std::string) "gaussian");
     SetInputParam("nystroem_method", true);
     SetInputParam("sampling", (std::string) "kmeans");
   
     mlpackMain();
   
     arma::mat output1 = IO::GetParam<arma::mat>("output");
   
     SetInputParam("input", x);
     SetInputParam("sampling", (std::string) "random");
   
     mlpackMain();
     arma::mat output2 = IO::GetParam<arma::mat>("output");
   
     SetInputParam("input", x);
     SetInputParam("sampling", (std::string) "ordered");
   
     mlpackMain();
     arma::mat output3 = IO::GetParam<arma::mat>("output");
   
     // The resulting matrices should be different.
     REQUIRE(arma::any(arma::vectorise(output1 != output2)));
     REQUIRE(arma::any(arma::vectorise(output2 != output3)));
     REQUIRE(arma::any(arma::vectorise(output1 != output3)));
   }


.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_main_tests_range_search_utils.hpp:

Program Listing for File range_search_utils.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_main_tests_range_search_utils.hpp>` (``/home/aakash/mlpack/src/mlpack/tests/main_tests/range_search_utils.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_TESTS_MAIN_TESTS_RANGE_SEARCH_TEST_UTILS_HPP
   #define MLPACK_TESTS_MAIN_TESTS_RANGE_SEARCH_TEST_UTILS_HPP
   
   #include <mlpack/methods/range_search/rs_model.hpp>
   #include <mlpack/core.hpp>
   #include <mlpack/core/util/mlpack_main.hpp>
   #include "../catch.hpp"
   
   inline std::string ModelToString(RSModel* model)
   {
     std::ostringstream oss;
     cereal::JSONOutputArchive oa(oss);
     oa(CEREAL_POINTER(model));
     return oss.str();
   }
   
   inline void CheckMatrices(std::vector<std::vector<double>>& vec1,
                             std::vector<std::vector<double>>& vec2,
                             const double tolerance = 1e-3)
   {
     REQUIRE(vec1.size()  == vec2.size());
     for (size_t i = 0; i < vec1.size(); ++i)
     {
       REQUIRE(vec1[i].size() == vec2[i].size());
       std::sort(vec1[i].begin(), vec1[i].end());
       std::sort(vec2[i].begin(), vec2[i].end());
       for (size_t j = 0 ; j < vec1[i].size(); ++j)
       {
         REQUIRE(vec1[i][j] == Approx(vec2[i][j]).epsilon(tolerance));
       }
     }
   }
   
   inline void CheckMatrices(std::vector<std::vector<size_t>>& vec1,
                             std::vector<std::vector<size_t>>& vec2)
   {
     REQUIRE(vec1.size()  == vec2.size());
     for (size_t i = 0; i < vec1.size(); ++i)
     {
       REQUIRE(vec1[i].size() == vec2[i].size());
       std::sort(vec1[i].begin(), vec1[i].end());
       std::sort(vec2[i].begin(), vec2[i].end());
       for (size_t j = 0; j < vec1[i].size(); ++j)
       {
         REQUIRE(vec1[i][j] == vec2[i][j]);
       }
     }
   }
   
   template<typename T>
   std::vector<std::vector<T>> ReadData(const std::string& filename)
   {
     std::ifstream ifs(filename);
     std::vector<std::vector<T>> table;
     std::string line;
     while (std::getline(ifs, line))
     {
       std::vector<T> numbers;
       T n;
       std::replace(line.begin(), line.end(), ',', ' ');
       std::istringstream stm(line);
       while (stm >> n)
         numbers.push_back(n);
       table.push_back(numbers);
     }
   
     return table;
   }
   
   #endif

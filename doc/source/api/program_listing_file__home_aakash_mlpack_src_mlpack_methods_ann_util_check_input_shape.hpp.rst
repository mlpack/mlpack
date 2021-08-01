
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_util_check_input_shape.hpp:

Program Listing for File check_input_shape.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_util_check_input_shape.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/util/check_input_shape.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_UTIL_CHECK_INPUT_SHAPE_HPP
   #define MLPACK_METHODS_ANN_UTIL_CHECK_INPUT_SHAPE_HPP
   
   #include <mlpack/methods/ann/visitor/input_shape_visitor.hpp>
   
   namespace mlpack {
   namespace ann {
   
   template<typename T>
   void CheckInputShape(const T& network,
                        const size_t inputShape,
                        const std::string& functionName)
   {
     for (size_t l = 0; l < network.size(); ++l)
     {
       size_t layerInShape = boost::apply_visitor(InShapeVisitor(), network[l]);
       if (layerInShape == 0)
       {
         continue;
       }
       else if (layerInShape == inputShape)
       {
         break;
       }
       else
       {
         std::string estr = functionName + ": the first layer of the network " +
             "expects " + std::to_string(layerInShape) + " elements, but the " +
             "input has " + std::to_string(inputShape) + " dimensions!";
         throw std::logic_error(estr);
       }
     }
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif

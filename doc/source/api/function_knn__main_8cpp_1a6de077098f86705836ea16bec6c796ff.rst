.. _exhale_function_knn__main_8cpp_1a6de077098f86705836ea16bec6c796ff:

Function BINDING_EXAMPLE("For, the following command will calculate the 5 nearest neighbors " "of each point in "+PRINT_DATASET("input")+" and store the distances " "in "+PRINT_DATASET("distances")+" and the neighbors in "+PRINT_DATASET("neighbors")+":" "\"+PRINT_CALL("knn", "k", 5, "reference", "input", "neighbors", "neighbors", "distances", "distances")+"\" "The output is organized such that row i and column j in the neighbors " "output matrix corresponds to the index of the point in the reference set " "which is the j 'th nearest neighbor from the point in the query set with " "index i. Row j and column i in the distances output matrix corresponds to" " the distance between those two points.")
=================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================

- Defined in :ref:`file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_knn_main.cpp`


Function Documentation
----------------------


.. doxygenfunction:: BINDING_EXAMPLE("For, the following command will calculate the 5 nearest neighbors " "of each point in "+PRINT_DATASET("input")+" and store the distances " "in "+PRINT_DATASET("distances")+" and the neighbors in "+PRINT_DATASET("neighbors")+":" "\"+PRINT_CALL("knn", "k", 5, "reference", "input", "neighbors", "neighbors", "distances", "distances")+"\" "The output is organized such that row i and column j in the neighbors " "output matrix corresponds to the index of the point in the reference set " "which is the j 'th nearest neighbor from the point in the query set with " "index i. Row j and column i in the distances output matrix corresponds to" " the distance between those two points.")

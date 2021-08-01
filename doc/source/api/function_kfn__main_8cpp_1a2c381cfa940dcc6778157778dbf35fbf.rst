.. _exhale_function_kfn__main_8cpp_1a2c381cfa940dcc6778157778dbf35fbf:

Function BINDING_EXAMPLE("For, the following will calculate the 5 furthest neighbors of each" "point in "+PRINT_DATASET("input")+" and store the distances in "+PRINT_DATASET("distances")+" and the neighbors in "+PRINT_DATASET("neighbors")+":" "\"+PRINT_CALL("kfn", "k", 5, "reference", "input", "distances", "distances", "neighbors", "neighbors")+"\" "The output files are organized such that row i and column j in the " "neighbors output matrix corresponds to the index of the point in the " "reference set which is the j 'th furthest neighbor from the point in the " "query set with index i. Row i and column j in the distances output file " "corresponds to the distance between those two points.")
============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================

- Defined in :ref:`file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_kfn_main.cpp`


Function Documentation
----------------------


.. doxygenfunction:: BINDING_EXAMPLE("For, the following will calculate the 5 furthest neighbors of each" "point in "+PRINT_DATASET("input")+" and store the distances in "+PRINT_DATASET("distances")+" and the neighbors in "+PRINT_DATASET("neighbors")+":" "\"+PRINT_CALL("kfn", "k", 5, "reference", "input", "distances", "distances", "neighbors", "neighbors")+"\" "The output files are organized such that row i and column j in the " "neighbors output matrix corresponds to the index of the point in the " "reference set which is the j 'th furthest neighbor from the point in the " "query set with index i. Row i and column j in the distances output file " "corresponds to the distance between those two points.")

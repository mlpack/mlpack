.. _exhale_function_random__forest__main_8cpp_1a7fae49eb5168e8af37971738080d28f9:

Function BINDING_EXAMPLE("For, to train a random forest with a minimum leaf size of 20 " "using 10 trees on the dataset contained in "+PRINT_DATASET("data")+"with labels "+PRINT_DATASET("labels")+", saving the output random " "forest to "+PRINT_MODEL("rf_model")+" and printing the training " ", one could call" "\"+PRINT_CALL("random_forest", "training", "data", "labels", "labels", "minimum_leaf_size", 20, "num_trees", 10, "output_model", "rf_model", "print_training_accuracy", true)+"\" ", to use that model to classify points in "+PRINT_DATASET("test_set")+" and print the test error given the labels "+PRINT_DATASET("test_labels")+" using that, while saving the " "predictions for each point to "+PRINT_DATASET("predictions")+", one " "could call " "\"+)
========================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================

- Defined in :ref:`file__home_aakash_mlpack_src_mlpack_methods_random_forest_random_forest_main.cpp`


Function Documentation
----------------------


.. doxygenfunction:: BINDING_EXAMPLE("For, to train a random forest with a minimum leaf size of 20 " "using 10 trees on the dataset contained in "+PRINT_DATASET("data")+"with labels "+PRINT_DATASET("labels")+", saving the output random " "forest to "+PRINT_MODEL("rf_model")+" and printing the training " ", one could call" "\"+PRINT_CALL("random_forest", "training", "data", "labels", "labels", "minimum_leaf_size", 20, "num_trees", 10, "output_model", "rf_model", "print_training_accuracy", true)+"\" ", to use that model to classify points in "+PRINT_DATASET("test_set")+" and print the test error given the labels "+PRINT_DATASET("test_labels")+" using that, while saving the " "predictions for each point to "+PRINT_DATASET("predictions")+", one " "could call " "\"+)

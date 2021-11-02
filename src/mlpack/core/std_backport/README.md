The files in this directory are taken from MNMLSTC Core 1.1.0 in order to 
backport features from C++ 17 standard library:

 * C++17 STL algorithms such as std::any and std::basic_string_view. 
 * Dependencies files that are used to implement these features.

These files are licensed under the Apache 2.0 License, available in LICENSE.txt
in this directory.

If you want a copy of mlpack without a dependence on the Apache License or 
without the backported version then you will need to

 * Remove this entire directory.
 * Remove the line "std_backport" from src/mlpack/core/CMakeLists.txt.
 * Use the C++17 standard by modifying the mlpack/CMakeLists.txt.

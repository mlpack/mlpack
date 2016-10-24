This repository contains an implementation of the hashing algorithm for
approximate furthest neighbor search detailed in the paper

"Approximate Furthest Neighbor in High Dimensions"
by Rasmus Pagh, Francesco Silverstri, Johan Siversten, and Matthew Skala
presented at SISAP 2015.

There is another implementation available here:
https://github.com/johanvts/FN-Implementations

but I wanted to re-implement this to ensure that I understood it correctly, and
so that I could get a better comparison.

This code is built using mlpack and Armadillo, so when you configure with CMake
you may have to specify the installation directory of mlpack and Armadillo, if
they are not already installed on the system.

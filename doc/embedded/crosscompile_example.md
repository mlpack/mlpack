## Crosscompile mlpack example on an embedded hardware

In this article, we explore, how to crosscompile and run an mlpack example code
on an embedded hardware such as Raspberry Pi 2. In our previous documentations,
we have explored how to run mlpack bindings such as kNN command line program on
a Raspberry PI. Please refer to that article first and follow the first
part on how to Setup cross-compilation toolchain, and then continue with this
article.

mlpack has an example repository that demonstrates a set of examples showing how
to use the library source code on different dataset and usecases including
embedded deployment. This tutorial basically explain our necessary CMake configurations
that are required to integrate with your local CMake to download mlpack
dependencies and cross compile the entire software.

If you have not used mlpack example repsoitory, I highly recommend to clone
this repository from this link to follow this tutorial:

```sh
git clone git@github.com:mlpack/examples.git
```

You can explore this repository and see the available example, we are
interested in the embedded ones. Please go to the embedded directory as we are
going to explore the source code of the CMakeLists.txt:



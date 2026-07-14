**Note: this wiki page is out of date, and probably isn't very useful for reference anymore.  It was written in 2010, and many things have changed since then.  So maybe it might be useful for you to read, but keep in mind that there are differences between what is written here and what is actually implemented in the CMake configuration now.  It may be more useful to refer to the current source code.**

Using CMake with **mlpack**
===========================

In early 2010, the build system has changed from the homebuilt `fl-build` system to [CMake](http://cmake.org/).

CMake is basically the equivalent to GNU Autotools.  With Autotools, you would type

```
$ ./configure
$ make
# make install
```

but with CMake it'll be just a little more complex, looking more like

```
$ mkdir build; cd build
$ cmake ../
$ make
# make install
```

At this time, our CMake setup has only been extensively tested when producing GNU Makefiles.  In the future it needs to be tested with more systems, like Visual Studio and so on.

In-Source / Out-Of-Source Builds
--------------------------------

I will reference this concept several times during this document, so if the words 'out-of-source build' are foreign to you, it may be helpful to read this section.

Traditionally, most projects have used the "in-source build" model.  That is, all the compilation and building is done right from the source directory you extracted.  But this has a big drawback: this puts `.o`s, `.a`s, and executables all over that source tree you extracted, and you have to reconfigure it each time you want to build.

If instead you make an entirely separate directory to build your project in, then you can keep all the build artifacts completely separate from the source code.  In addition, you can have multiple build directories, each with a different configuration.  For example, I personally keep `build_debug` and `build_nodebug` directories for convenience.

All of the code samples I give will be for the example of an out-of-source build.

Just Build It!
--------------

To build **mlpack** using the default options, first make your build directory:

```
$ mkdir build
$ cd build
```

Now, configure CMake.  The option we pass to CMake (`../`) is to tell it where the root of the source tree is (that is, where `mlpack/trunk/` is).

```
$ cmake ../
```

Finally, we can build **mlpack**.

```
$ make
```

Configuring **mlpack**
----------------------

So, after checking out the source, the first thing to do before building **mlpack** is to pick the configuration options you want.  This is the analog of `./configure` for those of you who have used autotools.  Here are the current options you can pass to the **mlpack** configuration:

* **`CMAKE_INSTALL_PREFIX`** - where the build artifacts will be installed if you type `make install`; default `/usr/local`
* **`DEBUG`** - compile with debugging symbols and no optimization (GCC `-g`); **default ON** (for non-releases)
* **`PROFILE`** - compile with profiling symbols (GCC `-pg`); **default ON** (for non-releases)

Also, the following options can be used to explicitly tell CMake where to find necessary libraries.  If you don't specify these options, CMake will search in default locations to try and find them.

* **`ARMADILLO_INCLUDE_DIR`** - directory where Armadillo headers are located (directory with `armadillo`)
* **`ARMADILLO_LIBRARY`** - path to libarmadillo.so
* **`PTHREADS_INCLUDE_DIR`** - include path of the Pthreads library
* **`PTHREADS_LIBRARY`** - location of the Pthreads library
* **`LAPACK_LIBRARIES`** - uncached list of libraries (using full path name) to link against to use LAPACK
* **`BLAS_LIBRARIES`** - uncached list of libraries (using full path name) to link against to use BLAS
* **`LIBXML2_INCLUDE_DIR`** - The LibXml2 include directory
* **`LIBXML2_LIBRARIES`** - The location of the LibXml2 library

These are not all the possible options; consult the [official CMake Documentation](http://cmake.org/cmake/help/documentation.html).

To specify an option, pass `-D <option>=<value>` to CMake.  For example, to compile without debugging or profiling symbols and specify that Armadillo is installed in `/home/ryan/armadillo/`, write

```
$ mkdir build
$ cd build
$ cmake -D DEBUG=OFF -D PROFILE=OFF -D ARMADILLO_INCLUDE_DIR=/home/ryan/armadillo/include/ -D ARMADILLO_LIBRARY=/home/ryan/armadillo/lib/libarmadillo.so ../
```

Here is another example, where we only change one option.

```
$ cd build
$ cmake -D CMAKE_INSTALL_PREFIX=/home/ryan/mlpack ../
```

So there, we have just changed the install prefix from the default of `/usr/local` to `/home/ryan/mlpack`.  This is useful for when you want to install **mlpack** locally (if you don't have root access on the system you are using).

You can also use an interactive CMake configuration tool, like `ccmake` (ncurses GUI).  I am sure there are more tools but I personally find the CMake command line interface sufficient and efficient.

It should also be noted that you can reconfigure a project at any time using the steps I just mentioned.  You can also have several existing configurations in different out-of-source build directories.  For instance, these commands would set up two build directories, one that would build without debugging or profiling symbols, and one with:

```
$ mkdir build_nodebug
$ cd build_nodebug
$ cmake -D DEBUG=OFF -D PROFILE=OFF ../
...
$ cd ..
$ mkdir build_sparse
$ cd build_sparse
$ cmake ../
...
```

Building **mlpack** (using GNU Makefiles)
-----------------------------------------

So far our tutorial here has assumed that you are using GNU Makefiles.  As noted earlier, more information will need to be added to explain how to do this with Visual Studio and other IDEs.

Once your build directory has been configured with CMake (and assuming you are in that directory), all you have to do is type `make` to build everything.  Or, you could type `make <target>` to make a specific target.  All the dependencies of the target you make will be built, and remember, at any time you can reconfigure and rebuild without needing to erase your build directory or anything like that.

Here are a couple examples of useful targets you might use:

```
$ make mlpack           # mlpack library (libmlpack.so)
$ make allknn           # the allknn executable from mlpack
$ make                  # mlpack library and all executables
```

You will also see that the directory hierarchy from the source tree is copied into the build tree.  If you just type `make` in the root build directory it will build everything, but it may take a while.

Hacking on **mlpack** with CMake
--------------------------------

So now, say that you are working on your amazing new machine learning method, but it doesn't compile right because you forgot to give one of your variables a type for some reason (e.g. [5466]).  So you make your change to the code in the source tree.  But you don't have to change the build tree or reconfigure the project or anything like that.  Just type `make <target>` or whatever you had been doing in the build tree and CMake will detect the changes, rebuild anything that needs to be rebuilt, and hopefully it will compile just fine.

Writing a CMakeLists.txt File
-----------------------------

You will notice that each directory has a `CMakeLists.txt` file; this is like a `build.py` file but for CMake, not for `fl-build`.  The easiest way to make one of these is probably to copy it from a nearby directory and then modify as necessary.  Here I will discuss the syntax for this file (and how it is actually very similar to `build.py`).  Below is the `CMakeLists.txt` file (as of [9802]) from `src/mlpack/methods/nca/`:

```
cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  nca.h
  nca_impl.h
  nca_softmax_error_function.h
  nca_softmax_error_function_impl.h
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all MLPACK sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_executable(nca
  nca_main.cc
)
target_link_libraries(nca
  mlpack
)

add_executable(nca_test
  nca_test.cc
)
target_link_libraries(nca_test
  mlpack
  boost_unit_test_framework
)
```

The first thing we do is set the `SOURCES` variable to include all of the files that will be compiled into the library (**mlpack**).  This is a very important thing to do and is one of the two things you must do.  Like the comment says, don't include the main executable or the test here (or anything that defines ```int main()```) -- that will cause things to fail.

The next stanza of lines is a bit of an ugly hack to add all of the files from `SOURCES` to a global variable called `MLPACK_SRCS` which is the full list of all **mlpack** sources.  This is used by the CMakeLists.txt at the top-level directory of the library in question.  This stanza will be the same in just about every project directory, so don't worry too much about it.

The other thing you have to do is define the executables produced in that directory, in the following format:

```
add_executable(<executable_name>
  executable_source_1.cc
  executable_source_2.cc
  ...
)
target_link_libraries(<executable_name>
  dependency_1
  dependency_2
)
```

This is fairly self-explanatory.  You will notice that in the old `build.py` files you had to specify which parts of FASTLIB or **mlpack** you were linking against.  Here, if you are linking against _anything_ in MLPACK, specify `mlpack` as a dependency.  Note that our NCA example above links against MLPACK; this is because those other source files in `src/mlpack/methods/nca/` are part of MLPACK.

A good process to make your own `CMakeLists.txt` is:

1. Copy `CMakeLists.txt` from a nearby directory.
1. Modify `SOURCES` to include all of the sources (the things you would put under `librule()` in `build.py`)
1. Make `add_executable()` and `target_link_libraries()` stanzas for any executables (things that would go under `binrule()` in `build.py`)

The last thing to do is to look in the `CMakeLists.txt` in the parent directory; it will have a list of directories that it recurses into.

**Make sure to add your directory to this list, otherwise CMake won't even attempt to compile it!**

Doing Complicated Things With CMake
-----------------------------------

If you don't have the time to learn CMake, you are likely best off just asking me (email: gth671b@mail.gatech.edu) what it is you want to do, because chances are, I've either done it before or can quickly point you in the right direction or just do it for you.

If you _do_ have the time to learn CMake, it is a neat system, and the documentation on their website is extensive.  Just make sure you use the right documentation (currently we are using CMake 2.8) since their API changes now and then.

Conclusion
----------

To wrap this all up, CMake is at least moderately cool.  It will help make **mlpack** more portable (since it supports generation of Visual Studio projects as output as well as other formats), and should keep development moving quickly.

To encourage you to agree with this, here is a picture of a kitten using CMake:


![http://existentialtype.net/wp-content/uploads/2007/05/im-in-ur-stackz-overflowing-ur-bufferz.jpg]
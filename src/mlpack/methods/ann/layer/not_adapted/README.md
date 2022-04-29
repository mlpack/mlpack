Layers in this directory were written with the old boost::visitor interface.  In
[#2777](https://github.com/mlpack/mlpack/pull/2777), we adapted each layer to
use inheritance instead.  However, time did not permit the adaptation of all
layers, and so remaining layers that have not yet been adapted are in this
directory.

The intention is that we will work our way through layers in this directory,
updating them to the new interface and re-enabling tests for them in separate,
follow-up PRs.  If you'd like to help out, you are more than welcome to!

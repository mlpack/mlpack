<object data="../img/pipeline-top-6.svg" type="image/svg+xml" id="pipeline-top">
</object>

# Deployment

Once a modeling pipeline is ready for deployment, it is easy to deploy mlpack
applications to a wide variety of settings due to its simple header-only nature.

See also the [examples repository](https://github.com/mlpack/examples/),
which contains a number of fully-working deployable example applications.

The pages below provide guidance for how to deploy mlpack to a variety of
relatively simple environments.

 * [***Compile an mlpack program***](compile.md): compile a standalone C++ program
   that uses mlpack.

 * [***Cross-compile to a Raspberry Pi***](../embedded/crosscompile_armv7.md):
   cross-compile an mlpack C++ application to an embedded or low-resource
   device.
   - See also the
     [cross-compilation setup page](../embedded/supported_boards.md).

 * [***Deploying mlpack on Windows***](deploy_windows.md): build a Windows
   application that uses mlpack.

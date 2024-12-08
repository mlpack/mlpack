# Developers

If you want to contribute to mlpack, or if you already are a regular contributor
to mlpack or a maintainer, the following pages may serve as useful documentation
about internal development processes, guidelines, and systems:

 * [Community](community.md): details of how the mlpack community operates and
   communicates, including *how to get involved*.

 * [Google Summer of Code](gsoc.md): advice on applying to mlpack for Google
   Summer of Code.

 * [CI/CD](ci.md): systems and servers involved in mlpack's continuous
   integration pipeline.

 * [Timers](timer.md): interface for timing bindings and other mlpack programs.

 * [Automatic binding system](bindings.md): design details and operation of
   mlpack's automatic binding generator, including how to add a new language.

 * [Writing a binding](iodoc.md): a tutorial on writing an mlpack binding that
   will automatically be compiled to any language mlpack has bindings for.

 * [Template policies](policies.md): documentation for standardized class
   interfaces used by mlpack algorithms.
   - [The ElemType policy](elemtype.md)
   - [The DistanceType policy](distances.md)
   - [The KernelType policy](kernels.md)
   - [The TreeType policy](trees.md)

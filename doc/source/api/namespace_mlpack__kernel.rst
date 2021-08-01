
.. _namespace_mlpack__kernel:

Namespace mlpack::kernel
========================


Kernel functions. 
 


.. contents:: Contents
   :local:
   :backlinks: none




Detailed Description
--------------------

This namespace contains kernel functions, which evaluate some kernel function :math:` K(x, y) ` for some arbitrary vectors :math:` x ` and :math:` y ` of the same dimension. The single restriction on the function :math:` K(x, y) ` is that it must satisfy Mercer's condition:
:math:`[ \int \int K(x, y) g(x) g(y) dx dy \ge 0 \`
for all square integrable functions :math:` g(x) `.
The kernels in this namespace all implement the KernelType policy. For more information, see :ref:`exhale_page_kernels`. 
 



Classes
-------


- :ref:`exhale_class_classmlpack_1_1kernel_1_1CauchyKernel`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1CosineDistance`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1EpanechnikovKernel`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1ExampleKernel`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1GaussianKernel`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1HyperbolicTangentKernel`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1KernelTraits`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1KernelTraits_3_01CauchyKernel_01_4`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1KernelTraits_3_01CosineDistance_01_4`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1KernelTraits_3_01EpanechnikovKernel_01_4`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1KernelTraits_3_01GaussianKernel_01_4`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1KernelTraits_3_01LaplacianKernel_01_4`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1KernelTraits_3_01SphericalKernel_01_4`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1KernelTraits_3_01TriangularKernel_01_4`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1KMeansSelection`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1LaplacianKernel`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1LinearKernel`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1NystroemMethod`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1OrderedSelection`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1PolynomialKernel`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1PSpectrumStringKernel`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1RandomSelection`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1SphericalKernel`

- :ref:`exhale_class_classmlpack_1_1kernel_1_1TriangularKernel`

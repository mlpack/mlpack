# This file should be incorporated into the main CMakeLists.txt when CMake 3.1
# becomes the minimum required version (we should at least wait until late 2016
# or early 2017 for this).
target_compile_features(mlpack PUBLIC
    cxx_decltype
    cxx_alias_templates
    cxx_auto_type
    cxx_lambdas
    cxx_constexpr
    cxx_rvalue_references
    cxx_static_assert
    cxx_template_template_parameters
    cxx_delegating_constructors
    cxx_variadic_templates
    cxx_nullptr
    cxx_noexcept
)

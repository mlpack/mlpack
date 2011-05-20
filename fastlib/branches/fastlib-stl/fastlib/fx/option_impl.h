#ifndef MLPACK_IO_OPTION_IMPL_H
#define MLPACK_IO_OPTION_IMPL_H

template<typename N>
Option<N>::Option(bool ignoreTemplate,
                N defaultValue,
                const char* identifier,
                const char* description,
                const char* parent,
                bool required) {
  if (ignoreTemplate)
    IO::Add(identifier, description, parent, required);
  else {
    IO::Add<N>(identifier, description, parent, required);
    IO::GetValue<N>(identifier) = defaultValue;
  }
}

template<typename N>
Option<N>::Option(N defaultValue,
                const char* identifier,
                const char* description,
                const char* parent,
                bool required) {
  IO::AddComplexType<N>(identifier, description, parent);
  IO::GetValue<N>(identifier) = defaultValue;
}
#endif 

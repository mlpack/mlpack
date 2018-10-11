#if defined(__cplusplus) || defined(c_plusplus)

extern "C"
{
#endif

void pca();

// This is just used to force Julia to load each .so in the order we need.
void loadSymbols();

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

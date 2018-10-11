#if defined(__cplusplus) || defined(c_plusplus)

extern "C"
{
#endif

void perceptron();

// This is just used to force Julia to load the .sos in the order we need.
void loadSymbols();

// Get the pointer to a binding-specific model type.
void* CLI_GetParamPerceptronModelPtr(const char* paramName);
// Set the pointer to a binding-specific model type.
void CLI_SetParamPerceptronModelPtr(const char* paramName, void* ptr);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#ifndef HMM_TESTING_H
#define HMM_TESTING_H

void GenerateAndTrainSequences(const char* transition_filename,
			       const char* emission_filename,
			       int n_sequences,
			       Matrix transition_matrices[],
			       Matrix emission_matrices[]);

void ComputeStationaryProbabilities(const Matrix &transition_matrix,
				    Vector* stationary_probabilities);

void SetToRange(int x[], int start, int end);

void RandPerm(int x[], int length);


#endif /* HMM_TESTING_H */

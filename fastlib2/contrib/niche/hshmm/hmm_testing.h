#ifndef HMM_TESTING_H
#define HMM_TESTING_H

void GenerateAndTrainSequences(const char* transition_filename,
			       const char* emission_filename,
			       int n_sequences,
			       ArrayList<Vector>* p_initial_probs_vectors,
			       ArrayList<Matrix>* p_transition_matrices,
			       ArrayList<Matrix>* p_emission_matrices,
			       int first_index);

void ComputeStationaryProbabilities(const Matrix &transition_matrix,
				    Vector* stationary_probabilities);

void SetToRange(int x[], int start, int end);

void RandPerm(int x[], int length);


#endif /* HMM_TESTING_H */

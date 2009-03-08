#ifndef HMM_TESTING_H
#define HMM_TESTING_H

void GenerateAndTrainSequences(const char* transition_filename,
			       const char* emission_filename,
			       int n_sequences,
			       ArrayList<Vector>* p_initial_probs_vectors,
			       ArrayList<Matrix>* p_transition_matrices,
			       ArrayList<Matrix>* p_emission_matrices,
			       int first_index);

void GetHMMSufficientStats(const Vector &observed_sequence,
			   int n_states, int n_symbols,
			   Vector* p_initial_probs_vector,
			   Matrix* p_transition_matrix,
			   Matrix* p_emission_matrix,
			   int max_iter_mmf, int max_rand_iter, double tol_mmf,
			   int max_iter_bw, double tol_bw);

void ObtainDataLearnHMMs(ArrayList<Vector> *p_initial_probs_vectors,
			 ArrayList<Matrix> *p_transition_matrices,
			 ArrayList<Matrix> *p_emission_matrices,
			 Matrix *p_training_data,
			 Matrix *p_test_data);

void SaveHMMs(const char* filename,
	      const ArrayList<Vector> &initial_probs_vectors,
	      const ArrayList<Matrix> &transition_matrices,
	      const ArrayList<Matrix> &emission_matrices,
	      const Matrix &training_data,
	      const Matrix &test_data);

void LoadHMMs(const char* filename,
	      ArrayList<Vector> *initial_probs_vectors,
	      ArrayList<Matrix> *transition_matrices,
	      ArrayList<Matrix> *emission_matrices,
	      Matrix *training_data,
	      Matrix *test_data);

void ComputeStationaryProbabilities(const Matrix &transition_matrix,
				    Vector* stationary_probabilities);

void SetToRange(int x[], int start, int end);

void RandPerm(int x[], int length);

int eval_loocv_svm(double c, int n_points, const ArrayList<index_t> &permutation, const Dataset& cv_set, datanode* svm_module, const Matrix &kernel_matrix, int *n_correct_class1, int *n_correct_class0);


#endif /* HMM_TESTING_H */

#ifndef TEST_ENGINE_H
#define TEST_ENGINE_H

#include "fastlib/fastlib.h"
#include "generative_mmk.h"
#include "latent_mmk.h"
#include "fisher_kernel.h"
#include "empirical_mmk.h"
#include "utils.h"
#include "contrib/niche/svm/smo.h"
#include "contrib/niche/svm/svm.h"

#define INSIDE_TEST_ENGINE_IMPL_H


void EmitResults(const Vector &c_set,
		 const GenVector<int> &n_correct_results,
		 int n_sequences);

void LoadCommonCSet(Vector* p_c_set);

void CreateIDLabelPairs(const GenVector<int> &labels,
			Matrix* p_id_label_pairs);

void TestHMMGenMMKClassification(const ArrayList<HMM<Multinomial> > &hmms,
				 const GenVector<int> &labels);

void TestHMMGenMMKClassification(const ArrayList<HMM<DiagGaussian> > &hmms,
				 const GenVector<int> &labels);

void TestHMMGenMMK2Classification(const ArrayList<HMM<Multinomial> > &hmms,
				  const GenVector<int> &labels);

void TestHMMGenMMK2Classification(const ArrayList<HMM<DiagGaussian> > &hmms,
				  const GenVector<int> &labels);

void TestHMMLatMMKClassification(const HMM<Multinomial> &hmm,
				 const ArrayList<GenMatrix<int> > &sequences,
				 const GenVector<int> &labels);

void TestHMMLatMMKClassificationKFold(int n_folds,
				      const ArrayList<HMM<Multinomial> > &kfold_hmms,
				      const ArrayList<GenMatrix<int> > &sequences,
				      const GenVector<int> &labels);

void TestHMMLatMMK2ClassificationKFold(int n_folds,
				       const ArrayList<HMM<Multinomial> > &kfold_hmms,
				       const ArrayList<GenMatrix<int> > &sequences,
				       const GenVector<int> &labels);

void TestHMMLatMMKClassificationKFold(int n_folds,
				      const ArrayList<HMM<Multinomial> > &kfold_exon_hmms,
				      const ArrayList<HMM<Multinomial> > &kfold_intron_hmms,
				      const ArrayList<GenMatrix<int> > &sequences,
				      const GenVector<int> &labels);

void TestHMMFisherKernelClassification(const HMM<Multinomial> &hmm,
				       const ArrayList<GenMatrix<int> > &sequences,
				       const GenVector<int> &labels);

template <typename TDistribution, typename T>
void TestHMMFisherKernelClassificationKFold(int n_folds,
					    const ArrayList<HMM<TDistribution> > &kfold_hmms,
					    const ArrayList<GenMatrix<T> > &sequences,
					    const GenVector<int> &labels);

template <typename TDistribution, typename T>
void TestHMMFisherKernelClassificationKFold(int n_folds,
					    const ArrayList<HMM<TDistribution> > &kfold_exon_hmms,
					    const ArrayList<HMM<TDistribution> > &kfold_intron_hmms,
					    const ArrayList<GenMatrix<T> > &sequences,
					    const GenVector<int> &labels);

void TestHMMBayesClassificationKFold(int n_folds,
				     const ArrayList<HMM<Multinomial> > &kfold_exon_hmms,
				     const ArrayList<HMM<Multinomial> > &kfold_intron_hmms,
				     const ArrayList<GenMatrix<int> > &sequences,
				     const GenVector<int> &labels);

void TestMarkovMMKClassification(int n_symbols,
				 const ArrayList<GenMatrix<int> > &sequences,
				 const GenVector<int> &labels);

void TestMarkovMMKClassification(const ArrayList<GenMatrix<double> > &sequences,
				 const GenVector<int> &labels);

int EvalKFoldSVM(double c, int n_points,
		 int n_folds,
		 const ArrayList<index_t> &permutation, const Dataset& cv_set,
		 datanode* svm_module, const Matrix &kernel_matrix,
		 int *n_correct_class1, int *n_correct_class0);

void SVMKFoldCV(const Matrix &id_label_pairs,
		const Matrix &kernel_matrix,
		const Vector &c_set);


#include "test_engine_impl.h"
#undef INSIDE_TEST_ENGINE_IMPL_H

#endif /* TEST_ENGINE_H */

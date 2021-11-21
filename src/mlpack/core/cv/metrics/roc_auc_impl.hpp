/**
 * @file core/cv/metrics/roc_auc_impl.hpp
 * @author Suvarsha Chennareddy
 *
 * The implementation of the class ROC AUC Score.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_ROC_AUC_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_ROC_AUC_IMPL_HPP

namespace mlpack {
namespace cv {

template<size_t classification>
template<typename MLAlgorithm, typename DataType, typename ResponsesType>
double ROC_AUC<classification>::Evaluate(MLAlgorithm& model,
                         const DataType& data,
                         const ResponsesType& labels)
{
    return Evaluate<classification>(model, data, labels);
}

/*
* Calculate the ROC AUC by splitting the area under the ROC graph into trapezoids.
* Only works for a binary setting.
*/
template<size_t classification>
template<size_t _classification,
    typename MLAlgorithm,
    typename DataType,
    typename ResponsesType,
    typename>
double ROC_AUC<classification>::Evaluate(MLAlgorithm& model,
    const DataType& data,
    const ResponsesType& labels)
{
    //Check dimensions
    util::CheckSameSizes(data, labels, "ROC_AUC::Evaluate()");

    arma::mat probabilities;


    // Taking Predicted Output (probabilities) from the model.
    model.Classify(data, probabilities);

    arma::Col<size_t> posLabels = arma::conv_to<arma::Col<size_t>>::from(
        (labels == 1));

    arma::Col<double> predictedResponses = arma::conv_to<arma::Col<double>>::from(
        probabilities.row(1));

    //Sort the predictions.
    arma::Col<double> sortedPredictedResponses = arma::sort(predictedResponses,
        "ascend", 0);

    //cout << sortedPredictedResponses << endl;
    //TP stands for True Positives and FP stands for False Positives.
    double TP1, FP1, TP2, FP2;

    double Area = 0;

    //Find the sum of the 1s in the responses vector. 
    //This value will be used later for the normalizing constant.
    double no_of_positive = arma::accu(posLabels);

    //The predictions themeselves will be used as thresholds. 
    //This is to find all possible (TP, FP) points.
    TP1 = arma::accu(
        (predictedResponses >= sortedPredictedResponses(0)) && posLabels
        );
    FP1 = arma::accu(
        (predictedResponses >= sortedPredictedResponses(0)) > posLabels
        );

    //Loop throught the thresholds
    for (size_t i = 1; i < (size_t)predictedResponses.n_elem; i++) {

        TP2 = arma::accu(
            (predictedResponses >= sortedPredictedResponses(i)) && posLabels
            );
        FP2 = arma::accu(
            (predictedResponses >= sortedPredictedResponses(i)) > posLabels
            );


        //Area of the trapezium formed by points (TP1, FP1) and (TP2, FP2).
        Area += (FP1 - FP2) * (TP1 + TP2) / 2;
        TP1 = TP2;
        FP1 = FP2;
    }
    //The sum of all those trapezium areas divided by the 
    //normalizing constant will give the ROC AUC.
    return Area / (no_of_positive * (posLabels.n_elem - no_of_positive));
}
/*
* For a multiclass setting, a generalized form of the AUC implemented 
* by Hand and Till is used.
* Source: https://link.springer.com/content/pdf/10.1023%2FA%3A1010920819831.pdf
*/
template<size_t classification>
template<size_t _classification,
         typename MLAlgorithm,
         typename DataType,
         typename ResponsesType,
         typename,
         typename>
double ROC_AUC<classification>::Evaluate(MLAlgorithm& model,
        const DataType& data,
        const ResponsesType& labels)
{
    //Check dimensions
    util::CheckSameSizes(data, labels, "ROC_AUC::Evaluate()");

    arma::Row<size_t> predictions;
    arma::mat probabilities;


    // Taking Predicted Output (probabilities) from the model.
    model.Classify(data, predictions,  probabilities);

    //cout << probabilities << endl;

    double AUC = 0;
    arma::uvec temp;
    size_t ni = 0;
    size_t nj = 0;
    for (size_t i = 0; i < (size_t)probabilities.n_rows; i++) {

        //The number of observations that belong to the class i.
        ni = arma::accu((labels == i));

        for (size_t j = i + 1; j < (size_t)probabilities.n_rows; j++) {

            //The number of observations that belong to the class j.
            nj = arma::accu((labels == j));

            //temp stores the sorted ranks of estimated probabilities
            //belonging to class i for all class i and j observations 
            sortRanks(probabilities, labels, i, j, ni, nj, temp);

            //Calculate the term A(i | j)
            AUC += (double) (sumOfRanks(temp, ni) - (ni*(ni + 1))/ 2)/(ni*nj);


            //temp stores the sorted ranks of estimated probabilities
            //belonging to class j for all class i and j observations
            sortRanks(probabilities, labels, j, i, nj, ni, temp);

            //Calculate the term A(j | i)
            AUC += (double) (sumOfRanks(temp, nj) - (nj*(nj + 1))/ 2)/(ni*nj);
            //Normaliz
        }

    }

    //Normalize with the constant c*(c-1) where c is number of classes
    return AUC/(probabilities.n_rows*(probabilities.n_rows - 1));
}
/*
* Method to find the vector of sorted ranks of estimated probabilities 
* belonging to class i for all class i and j observations 
*/
template<size_t classification>
template<typename ResponsesType>
void ROC_AUC<classification>::sortRanks(arma::mat probabilities, 
                                        ResponsesType labels,  
                                        size_t  i, size_t j, 
                                        size_t ni, size_t nj,
                                        arma::uvec& sortedRanks) {
    //Find the estimated probabilities of class i from the class i and j observations
    //and sort them
    arma::Row<double> i_classifications = arma::conv_to<arma::Row<double>>::from(
            (labels == i));

    arma::Row<double> j_classifications = arma::conv_to<arma::Row<double>>::from(
            (labels == j));

    arma::Row<double> i_probabilities = arma::sort(
            probabilities.row(i) % i_classifications, "ascend", 1);
    arma::Row<double> j_probabilities = arma::sort(
            probabilities.row(i) % j_classifications, "ascend", 1);

    //Get rid of zero terms
    i_probabilities.shed_cols(0, labels.n_elem - ni-1);
    j_probabilities.shed_cols(0, labels.n_elem - nj-1);

    //Get the sorted indexes
    sortedRanks = arma::sort_index(
        arma::join_rows(i_probabilities, j_probabilities),
        "ascend");
}

/**
* Method to calculate the sum of ranks of estimated probabilities of class i 
* from observations that belong to class i (S term)
*/
template<size_t classification>
size_t ROC_AUC<classification>::sumOfRanks(arma::uvec sortedRanks, size_t n) {
    size_t s = 0;
    for (size_t i = 0; i < (size_t)sortedRanks.n_elem; i++) {
        if (sortedRanks(i) < n) {
            s += i+1;
        }
    }
    return s;
}
} // namespace cv
} // namespace mlpack

#endif

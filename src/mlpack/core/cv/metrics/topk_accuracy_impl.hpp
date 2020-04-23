/**
 * @file topk_accuracy_impl.hpp
 * @author Arunav Shandeelya
 * 
 * The implementation of class topK_accuracy_score
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_CV_METRICS_TOPK_ACCURACY_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_TOPK_ACCURACY_IMPL_HPP

namespace mlpack {
namespace cv {
    template<typename MLAlgorithm,typename DataType,typename ResponseType,typename TopK>
    double TopK_Accuracy::Evaluate(MLAlgorithm& model,
                                    const DataType& data,
                                    const arma::Row<size_t>& labels,
                                    const TopK& k)

    {
        if(data.n_cols! = labels.n_cols)
        {
            std::ostringstream fp;
            fp << "topK_accuracy_score::Evaluate(): number of inputs ("<< data.n_cols <<")" 
            << "does not matches with number of target labels ("<< labels.n_cols <<")" 
            <<std::endl;
        throw std::invalid_argument(fp.str());
        }
        arma::Row<size_t> predictedLabels;
        // Taking Classification output from the model.
        model.Classify(data, predictedLabels)
        // Shape of the predicted values.
        double predicted_class = predictedLabels.size();
        // Top 'K' label prediction class. 
        if (k<predicted_class[1]){
            const int idx = predicted_class[1] - k - 1;
        }
        else:
        {
            std::ostringstream fs;
            fs <<"There are less number of class for topK accuracy score" <<std::endl;
        }
        const int idx = predicted_class[1] - k - 1;
        const float c = 0;
        double srt = sort_index(predictedLabels);  

        for(i=1;i<predicted_class[0];i++)
        {
            if(labels[i]!= srt[i, idx+1:])
            {
                c += 1;
            }
        }
        // Accuracy Score of top k predicted class labels.
        return (double) c / predicted_class[0]
    }

} // namespace cv
} // namespace mlpack

#endif


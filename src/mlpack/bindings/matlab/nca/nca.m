function result = nca(dataPoints, labels)
%Neighborhood Components Analysis (NCA)
%
%  This program implements Neighborhood Components Analysis, both a linear
%  dimensionality reduction technique and a distance learning technique.  The
%  method seeks to improve k-nearest-neighbor classification on a dataset by
%  scaling the dimensions.  The method is nonparametric, and does not require a
%  value of k.  It works by using stochastic ("soft") neighbor assignments and
%  using optimization techniques over the gradient of the accuracy of the
%  neighbor assignments.
%
%  To work, this algorithm needs labeled data.  It can be given as the last row
%  of the input dataset (--input_file), or alternatively in a separate file
%  (--labels_file).
%
%Parameters:
% dataPoints - Input dataset to run NCA on.
% labels     - Labels for input dataset.

% interfacing with mlpack. transposing to machine learning standards.
result = mex_nca(dataPoints', labels);
result = result';



function assignments = emst(dataPoints, clusters, varargin)
%K-Means Clustering
%
%  This program performs K-Means clustering on the given dataset, storing the
%  learned cluster assignments either as a column of labels in the file
%  containing the input dataset or in a separate file.  Empty clusters are not
%  allowed by default; when a cluster becomes empty, the point furthest from the
%  centroid of the cluster with maximum variance is taken to fill that cluster.

% a parser for the inputs
p = inputParser;
p.addParamValue('allow_empty_clusters', false, @(x) (x == true) || (x == false));
p.addParamValue('fast_kmeans', false, @(x) (x == true) || (x == false));
p.addParamValue('max_iterations', 1000, @isscalar);
p.addParamValue('overclustering', 1, @isscalar);
p.addParamValue('seed', 0, @isscalar);

% parsing the varargin options
p.parse(varargin{:});
parsed = p.Results;

% interfacing with mlpack. transposing to machine learning standards.
assignments = mex_kmeans(dataPoints', clusters, parsed.max_iterations, ...
	parsed.overclustering, parsed.allow_empty_clusters, ...
	parsed.fast_kmeans, parsed.seed);

assignments = assignments + 1; % changing to matlab indexing


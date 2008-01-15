function [centered_data] = center(data);
% USAGE: center(data)
% data is a d by n matrix

n = size(data,2);

centered_data = data - repmat(mean(data, 2), 1, n);

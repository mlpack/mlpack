function X = loadImages(dirname)
listing = dir([dirname '/*.gif']);
n = length(listing);
X = [];
for i=1:n
    x_i = imread([dirname '/' listing(i).name]);
    x_i = x_i(:);
    X = [X x_i];
end
X = double(X)/256;
end
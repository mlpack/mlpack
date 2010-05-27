function [images, images_line] = convert_to_images(X, n_rows, n_cols)
X(X > 1) = 1;
X = uint8(X*256);
[m, n] = size(X);
images = cell(n,1);
images_line = ones(n_rows, 1)*30;
for i = 1:n
    images{i} = reshape(X(:, i), n_rows, n_cols);
    images_line = [images_line images{i} ones(n_rows, 1)*30];
end
end
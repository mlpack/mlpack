prec = 0.0001;

pca_coef = getcoef(pca_results.harmfd);
basis_points = eval_basis(prec:prec:.5, mybasis);

pca_points = basis_points * pca_coef;

data_coef = getcoef(myfd_train);
data_points = basis_points * data_coef;



%some_points = pca_points(:,1) .^ 2;
some_points = pca_points(:,1) .* data_points(:,1);

pp = spline(prec:prec:.5, some_points);

a = quad(@ppval, 0, .5, [], [], pp)


some_points = pca_points(:,2) .* data_points(:,1);

pp = spline(prec:prec:.5, some_points);

a = quad(@ppval, 0, .5, [], [], pp)

some_points = pca_points(:,3) .* data_points(:,1);

pp = spline(prec:prec:.5, some_points);

a = quad(@ppval, 0, .5, [], [], pp)



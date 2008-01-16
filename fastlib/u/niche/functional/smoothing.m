p = 8;

indices = 1:200:1000;

t_coarse = t(indices);

x = data(:,1);

coarse_x = x(indices);

mybasis = create_bspline_basis([0 1], p, 4);
basis_curves = eval_basis(t, mybasis);

myfd_x = data2fd(x, t, mybasis);
x_coef = getcoef(myfd_x);
x_curves = basis_curves * x_coef;

myfd_coarse_x = data2fd(coarse_x, t_coarse, mybasis);
coarse_x_coef = getcoef(myfd_coarse_x);
coarse_x_curves = basis_curves * coarse_x_coef;

lambda = 1e-4;
myfdPar = fdPar(mybasis, 2, lambda); 
myfd_smooth_x = smooth_fd(myfd_coarse_x, myfdPar);
smooth_x_coef = getcoef(myfd_smooth_x);
smooth_x_curves = basis_curves * smooth_x_coef;


figure(1);
clf;
hold on;
plot(t, x_curves(:,1), 'b');
plot(t, coarse_x_curves(:,1), 'r');
plot(t, smooth_x_curves(:,1), 'g');

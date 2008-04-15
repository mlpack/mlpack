%Comparison of the gaussian and the erf kernel


function[]=compare_kernels()
x=[-2:0.1:2];
h=0.5;
for i=1:length(x)
    y1=sqrt((1/6.28))*power('e',-0.5*power(x(i)/h,2))/h;
    y1_vec(i)=y1;
    y2=(erf(x(i)/(h*sqrt(2))+0.5/sqrt(2.0))-erf(x(i)/(h*sqrt(2))-0.5/sqrt(2)))/2*h;
    y2_vec(i)=y2;
end

[number,cols]=size(x);
'number_of_elements is'

%plot_matrix(1:cols,1:1)=x;
plot_matrix(1:cols,1:1)=y1_vec;
plot_matrix(1:cols,2:2)=y2_vec;

plot(x,plot_matrix);
'normal function is'
y1_vec

'erf function is'
y2_vec
legend('nomral','erf');

h=0.6945
z=((erf(1/(h*sqrt(2))+0.5/sqrt(2.0))-erf(1/(h*sqrt(2))-0.5/sqrt(2)))/(4*h)+...+
    (erf(1/(h*sqrt(2))+0.5/sqrt(2.0))-erf(1/(h*sqrt(2))-0.5/sqrt(2)))/(4*h));
'z is'
z
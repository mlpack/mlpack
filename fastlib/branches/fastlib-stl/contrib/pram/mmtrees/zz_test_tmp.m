
n=100000; % number of point
m=2; % demetion

% geration with randn:
v=(2*rand(m,n)-1);
for nc=1:n
    v2=v(:,nc)'*v(:,nc);
    v(:,nc)=v(:,nc)/sqrt(v2);
end

al=atan2(v(2,:),v(1,:));

al=180*al/pi;

subplot(2,1,1);
hist(al,n/500);
ylim([0 1000]);
title('generation with rand');
xlabel('degrees');


% generation with random_unit_vector:
v=random_unit_vector(m,n);


al=atan2(v(2,:),v(1,:));
al=180*al/pi;

subplot(2,1,2);
 hist(al,n/500);
 ylim([0 1000]);
xlabel('degrees'); 
title('generation with randn');
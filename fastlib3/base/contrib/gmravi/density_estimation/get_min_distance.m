function[]= get_min_distance()
data1=csvread('points_mog1_test.txt');
data2=csvread('points_mog1_train.txt');
[n_rows1,n_cols1]=size(data1);
[n_rows2,n_cols2]=size(data2);
min_dist=100000;
for i=1:n_rows1
    for j=1:n_rows2
        dist=abs(data1(i,1)-data2(j,1));
        if(dist<min_dist)
            min_dist=dist;
        end
    end
end
display(min_dist);
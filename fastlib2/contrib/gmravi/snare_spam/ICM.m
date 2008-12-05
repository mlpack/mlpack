function[G_mat]=ICM(num_rows,epsilon)
Q_mat=GetQMatrix()
G_mat=sparse(zeros(num_rows,num_rows));

for i=1:num_rows
    display('Before changes.G matrix is');
    display(full(G_mat));

    for j=i:num_rows
        G_mat(j,j)=GetDiagonalElement(Q_mat,j); %Gets Q(j,j)
        for k=1:i-1
            G_mat(j,j)=G_mat(j,j)-(G_mat(j,k)*G_mat(j,k));
        end
    end
    
    display('After changes.G matrix is');
    display(full(G_mat));
   
    if(SumDiagonalElements(G_mat,i)>epsilon)
        j_star=FindIndexOfMaxDiagonalElement(G_mat,i);
        display(j_star);
       % G_mat(j_star:num_rows,i:i)=...
            GetSubMatrixOfQ(Q_mat,j_star,num_rows,j_star,j_star);
        
        display('After copy')
        display(full(G_mat));

        temp=G_mat(j_star:j_star,1:i-1);
        G_mat(j_star:j_star,1:i)=G_mat(i:i,1:i-1);
        G_mat(i:i,1:i-1)=temp;
        
        display('After swap')
        display(full(G_mat));

%         for j=1:i-1
%             G_mat(i+1:num_rows,i:i)=G_mat(i+1:num_rows,i:i)-...
%                 (G_mat(i+1:num_rows,j:j)*G_mat(i,j));
%         end
        G_mat(i+1:num_rows,i) = G_mat(i+1:num_rows,i) - G_mat(i+1:num_rows,1:i-1) * G_mat(i,1:i-1)';
        
        display('Before divindin');
        display(full(G_mat));
        G_mat(i,i)=sqrt(G_mat(i,i));
        
        G_mat(i+1:num_rows,i:i)=G_mat(i+1:num_rows,i:i)/G_mat(i,i);
        
    else
        k=i-1;
        break;
    end 
end

G_mat=G_mat(1:end,1:k);
G_mat=full(G_mat);
display(G_mat);


function[sub]=GetSubMatrixOfQ(Q_mat,start_row,end_row,start_col,end_col)
sub=Q_mat(start_row:end_row,start_col:end_col); 
display('submatri xis');
display(sub);

        

function[Q_mat]=GetQMatrix()
Q_mat=[1,2,3;2,4,6;3,6,9];
%Q_mat=[14,32;32,77];
%Q_mat=[10,14;14,20];

function[val]=GetDiagonalElement(Q_mat,row_num) 
val=Q_mat(row_num,row_num);

function[val]=SumDiagonalElements(G_mat,start_pos)
dg=diag(G_mat);
dg=dg(start_pos:end);
val=sum(dg);

function[ind]=FindIndexOfMaxDiagonalElement(G_mat,start_pos)
dg=diag(G_mat);
dg=dg(start_pos:end);
[y, index] = max(dg);
ind=index+start_pos-1;
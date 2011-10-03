from cvxopt import *

# First we need some data points and labels
#points:
# randomly generated for now
n_dims = 2;
n_points = 60;

R = matrix([[cos(pi/4), -sin(pi/4)], [sin(pi/4), cos(pi/4)]]);

# generate positive points
Xpos = uniform(n_dims, n_points / 2, -1.0, 1.0);
Xpos[0,:] *= 2;
Xpos = R * Xpos;
Xpos += 2;

# generate negative points
Xneg = uniform(n_dims, n_points / 2, -1.0, 1.0);
Xneg[0,:] *= 2;
Xneg = R * Xneg;
Xneg -= 2;

# put points together
X = matrix([[Xpos], [Xneg]]);

# create a perfect linear separator
w = matrix([cos(pi/4), sin(pi/4)]);
# label data
y = w.T * X;
y = sign(y)[0];


# min  c^T x
# s.t. G x + s = h
#      A x = b
#      s >= 0

G = matrix(0.0, (1 + n_points * (1 + n_dims), n_dims));
h = matrix(0.0, (1 + n_points * (1 + n_dims), 1));

G[0,:] = -y[0] * X[:,0].T;
h[0] = -1e-10;

gamma = min(abs(w.T * X));

cur_row = 1;

for i in range(0,n_points):
    print "i = ",i
    G[cur_row,:] = -y[i] * X[:,i].T;
    cur_row += 1;
    for j in range(0, n_dims):
        print "j = ",j
        # write row
        # the next line needs to be changed later to handle the case of missing dimensions
        G[cur_row,j] = -gamma;
        cur_row += 1;

c = matrix(0.0, (n_dims, 1));

dims = {'l': 1, 'q': [1 + n_dims] * n_points, 's': []};

sol =  solvers.conelp(c, G, h, dims);
w_hat = sol['x'];



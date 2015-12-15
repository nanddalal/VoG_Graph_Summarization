a = [[ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
     [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
     [ 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
     [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
     [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.],
     [ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.],
     [ 1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.],
     [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.]];

cs = 50;
a = ones(cs) - eye(cs);

tnn = 12000;

[fc, ~] = mdlCostAsfANDnClique(a,tnn);
[st, ~, ~] = mdlCostAsStar(a,[],tnn);
[bc, nb, ~, ~] = mdlCostAsBCorNB(a,tnn);
cost_notEnc = compute_encodingCost( 'err', 0, 0, [nnz(a) cs^2-nnz(a)]);

disp([fc, st, bc, nb, cost_notEnc])

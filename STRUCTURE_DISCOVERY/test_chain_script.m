N_total = 40;
Asmall = [];
Asmall = [Asmall; [0 1 1 0 0 0 0 0 0 1]];
Asmall = [Asmall; [1 0 0 0 0 0 1 0 0 0]];
Asmall = [Asmall; [1 0 0 1 0 0 0 0 0 0]];
Asmall = [Asmall; [0 0 1 0 1 0 0 0 0 0]];
Asmall = [Asmall; [0 0 0 1 0 1 0 0 1 0]];
Asmall = [Asmall; [0 0 0 0 1 0 0 0 0 0]];
Asmall = [Asmall; [0 1 0 0 0 0 0 1 0 0]];
Asmall = [Asmall; [0 0 0 0 0 0 1 0 0 0]];
Asmall = [Asmall; [0 0 0 0 1 0 0 0 0 0]];
Asmall = [Asmall; [1 0 0 0 0 0 0 0 0 0]];

[ MDLcost, chainExt ] = mdlCostAsChain( Asmall, N_total );




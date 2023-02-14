% Do a Li wf optimization in a basis of shifted correlated Gaussians



#Mass = [0.500039104068 3.91040678074e-5 3.91040678074e-5;
#        3.91040678074e-5 0.500039104068 3.91040678074e-5;
#        3.91040678074e-5 3.91040678074e-5 0.500039104068];

Mass = [0.5 0.0 0.0
        0.0 0.5 0.0
        0.0 0.0 0.5];

Charge = [-3 1 1 -3 1 -3]';

% symmetry projection terms
%Sym = zeros(3,3,6)
% (1)(2)(3)
Sym(:,:,1) = [1 0 0; 0 1 0; 0 0 1];
% (12)
Sym(:,:,2) = [0 1 0; 1 0 0; 0 0 1];
% (13)
Sym(:,:,3) = [0 0 1; 0 1 0; 1 0 0];
% (23)
Sym(:,:,4) = [1 0 0; 0 0 1; 0 1 0];
% (123)
Sym(:,:,5) = [0 1 0; 0 0 1; 1 0 0];
% (132)
Sym(:,:,6) = [0 0 1; 1 0 0; 0 1 0];

% coeff's
symc = [4.0 4.0 -2.0 -2.0 -2.0 -2.0]';

n=3;
nb=8;

xvechL=[
1.6210e+00
-2.1504e-01
 9.0755e-01
 9.7866e-01
-2.8418e-01
-3.5286e+00
-3.3045e+00
-4.5036e+00
-3.2116e-01
-7.1901e-02
 1.5167e+00
-8.4489e-01
-2.1377e-01
-3.6127e-03
-5.3774e-03
-2.1263e+00
-2.5191e-01
 2.1235e+00
-2.1396e-01
-1.4084e-03
-1.0092e-02
 4.5349e+00
 9.4837e-03
 1.1225e+00
-2.1315e-01
 5.8451e-02
-4.9410e-03
 5.0853e+00
 7.3332e-01
 5.0672e+00
-2.1589e-01
-6.8986e-03
-1.4310e-02
 1.5979e+00
 3.3946e-02
-8.7965e-01
-1.1121e+00
-2.1903e-03
-4.6925e-02
 2.1457e-01
 3.3045e-03
 4.5120e+00
-2.1423e-01
-1.6493e-02
-2.3429e-03
-8.6715e-01
-6.7070e-02
 1.5998e+00
 ];


evec = [
-6.0460e-02
   7.7708e-05
   1.6152e+00
   9.5443e-01
   1.1771e-01
   3.2196e+00
   9.6344e-01
   3.1398e+00

  ];


x1 = [xvechL' evec']';

cg_energyrc(x1,n,nb,Mass,Charge,Sym,symc)

%global n nb Mass Charge Sym symc

%  global n nb Mass Charge Sym symc;
  f = @(x)cg_energyrc(x,n,nb,Mass,Charge,Sym,symc);

%f(x1)
%options(1)=1;
options = optimset('Display','iter');
%x0 = x1;
fminsearch(f,x1,options)
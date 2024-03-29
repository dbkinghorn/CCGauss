% Do a Li wf optimization in a basis of shifted correlated Gaussians



Mass = [0.500039104068 3.91040678074e-5 3.91040678074e-5;
        3.91040678074e-5 0.500039104068 3.91040678074e-5;
        3.91040678074e-5 3.91040678074e-5 0.500039104068];

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
2.5506561
-0.2682935
-1.6094056
 0.7783217
-0.9078165
-2.0758450
-2.3378436
-2.0308924
 0.6521045
-0.0850259
 1.3587241
-0.6734138
-0.1980076
 0.1026624
 0.0765663
-2.0217744
-0.2719491
 1.8464342
-0.1925625
-0.0844815
-0.0667643
 4.3249007
 0.0043200
 1.0781679
-0.1968378
-0.2357974
 0.5343730
 3.0986425
 0.3096986
 2.8633729
-0.1974198
-0.0787332
-0.0269395
 1.6174593
 0.0131678
-0.8146160
-1.0902467
-0.0075315
-0.1055016
 0.1908637
 0.1633921
 4.0378356
-0.1972251
-0.0165167
-0.0292860
-0.9158674
-0.0527130
 1.4637889

 ];


evec = [
-0.0464404
 0.0057956
 1.5830024
 0.9582031
 0.3062538
 2.4154218
 1.1493228
 3.3468503
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

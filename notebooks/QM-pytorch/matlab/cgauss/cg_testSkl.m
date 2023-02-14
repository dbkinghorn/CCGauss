% simple test for simple correlated gaussian Skl code
n = 3;
vechLk = [  1.00000039208682
   0.02548044275764261
   0.3525161612610669
   1.6669144815242515
   0.9630555318946559
   1.8382882034659822 ]';

vechLl = [   1.3353550436464964
   0.9153272033682132
   0.7958636766525028
   1.8326931436447955
   0.3450426931160630
   1.8711839323167831 ]';


sym = [0 0 1;0 1 0;1 0 0];
%sym = [1 0 0;0 1 0;0 0 1];

Mass = [5.446170e-4 2.723085077e-4 2.723085077e-4;
        2.723085077e-4 .5002723085 2.723085077e-4;
        2.723085077e-4 2.723085077e-4 .5002723085];


vecQ = [1 -1 -1 -1 1 -1]';

n
vechLk
vechLl
sym
Mass
vecQ

[skl, tkl, vkl, hkl] = cg_matel(n,vechLk,vechLl,sym,Mass,vecQ);

skl

tkl

vkl

hkl

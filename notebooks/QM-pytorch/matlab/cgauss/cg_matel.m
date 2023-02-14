function [skl, tkl, vkl, hkl] = cg_matel(n,vechLk,vechLl,sym,Mass,vecQ)

% matel: returnes symmetry projected matrix elements in a
% basis of simple correlated gaussians fkl = exp[-r'(Lk*Ll' kron I3)r]

% n:		the number of psuedo particles i.e.  N-1
% vechLk:	nonlinear exponent parameters n(n+1)/2 x 1
% vechLl:
%
%
% sym:		symmetry projection matrix
% Mass:		mass matrix for kinetic energy
% vecQ:		charge products for potential energy

% initialize arrays
Lk=zeros(n,n);
Ll=zeros(n,n);
Ak=zeros(n,n);
Al=zeros(n,n);
Akl=zeros(n,n);
invAkl=zeros(n,n);
invAk=zeros(n,n);
invAl=zeros(n,n);

% build Lk and Ll

count=0;
for j=1:n
  for i=j:n
    count=count+1;
    Lk(i,j) = vechLk(count);
    Ll(i,j) = vechLl(count);
  end
end

% apply symmetry projection

PLl = sym'*Ll;

% build Ak, Al, Akl, invAkl, invAk, invAl

Ak = Lk*Lk';
Al = PLl*PLl'
Akl = Ak+Al;
invAkl = inv(Akl);
invAk = inv(Ak);
invAl = inv(Al);

% Overlap: (normalized)
skl = 2^(3*n/2) * sqrt( (abs(det(Lk))*abs(det(Ll))/det(Akl) )^3 );

dsk1 = 3/2 * skl * (inv(Lk)' - 2 * invAkl*Lk)
dsk2 = 3/2 * skl * (diag(1./diag(Lk)) - 2 * invAkl*Lk)
dsl1 = 3/2 * skl * (inv(Ll)' - 2 * sym*invAkl*PLl)
dsl2 = 3/2 * skl * (diag(1./diag(Ll)) - 2 * sym*invAkl*PLl)
vech(dsl2)
% kinetic energy

tkl = skl*(6*trace(Mass*Ak*invAkl*Al));

% potential energy

% 1/rij i~=j
for j=1:n-1
  for i=j+1:n
    tmp2 = invAkl(i,i) + invAkl(j,j) - 2*invAkl(i,j);
    RIJ(i,j) = 2/sqrt(pi) * skl/sqrt(tmp2);
  end
end

% 1/rij i=j
for i=1:n
  RIJ(i,i) = 2/sqrt(pi) * skl/sqrt(invAkl(i,i));
end
RIJ
vkl = 0;
count=0;
for j=1:n
  for i=j:n
    count=count+1;
    vkl = vkl + vecQ(count)*RIJ(i,j);
  end
end

% Hamiltonian matrix element
hkl = tkl + vkl;

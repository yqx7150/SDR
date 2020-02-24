function A = A_fhp3D(z, S,n1,n2,n3)
z=double(z); S = double(S);
%[n1,n2,n3] = size(z);
z = reshape(z,n1,n2,n3);
p=1/sqrt(n1*n2)*fft2(z);
A = p(S);

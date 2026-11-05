function validateVectorParameters(p,T)
% Independent CIR factors, am=0, with finite discount/half/forward moments.
assert(ismatrix(p) && size(p,2)==9 && ~isempty(p) && all(isfinite(p),'all'));
assert(all(p(:,1)>0 & p(:,2)>=0 & p(:,3)>0 & abs(p(:,4))<=1 & p(:,5)>=0));
assert(all(p(:,6)==0),'This implementation supports the am=0 model.');
for i=1:size(p,1)
 q=p(i,:);k=q(1);s=q(3);rho=q(4);a=q(7);
 for moment=[0 .5 1]
  c=k-moment*rho*s*a;
  b=(-moment^2+moment)*a^2-2*moment*(q(9)-q(8))+2*q(9);
  delta=c*c+s*s*b;explosion=inf;
  if b<0
   if delta<0
    d=sqrt(-delta);explosion=2*(pi/2+atan(c/d))/d;
   elseif c<0
    if delta==0,explosion=-2/c;
    else,d=sqrt(delta);explosion=log((c-d)/(c+d))/d;end
   end
  end
  assert(T<explosion,'Vector:MomentDomain','Required discounted moment explodes before the horizon.');
 end
end
end

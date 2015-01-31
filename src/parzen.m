%harrvec1 = [0.01:5:1000]';
%harrvec2 = [0.01:5:1000]';
%harrvec  = [harrvec1 harrvec2];
%sizeHarr = size(harrvec);
%sizeHarr = sizeHarr(:,1);

n = 300;
harr = [1 1];

ix11 = randperm(100);
ix12 = randperm(100);
ix2  = randperm(100);
ix66 = randperm(66);
 
%for m = 1:(sizeHarr(:,1))
set11 = r11;
set12 = r12;
set11 = set11(ix11,:);
set12 = set12(ix12,:);
train = [set11([1:33],:); set12([1:33],:)];
test1 = [set11([34:100],:); set12([34:100],:)];
train = train(ix66,:);


outpt1 = ones(1,n);
for k = 1:n
    outpt1(1,k) = bivar(r(k,:), train, harr, 66);
end

set2 = r2;
train2 = set2(ix2,:);
test2  = set2([34:100],:);

outpt2 = ones(1,n);
for k = 1:n
    outpt2(1,k) = bivar(r(k,:), train2([1:33],:), harr,33);
end


%Computando P(wi | xk)
Pw1 = ones(1,n);
Pw2 = ones(1,n);
for k = 1:n
    Pw1(1,k) = (outpt1(1,k)*(2/3))/(outpt1(1,k)*(2/3) + outpt2(1,k)*(1/3));
    Pw2(1,k) = (outpt2(1,k)*(1/3))/(outpt1(1,k)*(2/3) + outpt2(1,k)*(1/3));
end

decida = ones(1,n);
% Classificando...
for k = 1:n
    if(Pw1(1,k) > Pw2(1,k))
        decida(1,k) = 1;
    else
        decida(1,k) = 0;
    end
end
%end
%harrvec1 = [0.01:5:1000]';
%harrvec2 = [0.01:5:1000]';
%harrvec  = [harrvec1 harrvec2];
%sizeHarr = size(harrvec);
%sizeHarr = sizeHarr(:,1);

n = 300;
harr = [10 10];


%for m = 1:(sizeHarr(:,1))
train = [r11; r12];
outpt1 = ones(1,n);
for k = 1:n
    outpt1(1,k) = bivar(r(k,:), train, harr);
end
    
train2 = r2;
outpt2 = ones(1,n);
for k = 1:n
    outpt2(1,k) = bivar(r(k,:), train2, harr);
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
function pdf = bivar(teste, train, harr)


n = size(train,1);
harr = (1/sqrt(n)).*harr;
temp = 0;
for i = 1:n
    temp = temp + multi(teste, train(:,1), harr);
end
pdf = temp/n;

train = [r11; r12];
harr = [0.6 0.6];
n = 200;
outpt = ones(n,n);

for k = 1:n
	  for l = 1:n
		    outpt(l,k) = bivar([l k], train, harr);
          end
end

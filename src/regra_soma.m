function classe = regra_soma(p1a, p1b, p1c, p2a, p2b, p2c)

classe = [];

for i = 1:300
    if((-2)*(2/3) + (p1a(i) + p1b(i) + p1c(i)) > (-2)*(1/3) + (p2a(i) + p2b(i) + p2c(i)))
        classe = [classe 1];
    else
        classe = [classe 0];
    end
end

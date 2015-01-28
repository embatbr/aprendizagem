function result = multi(teste, train, harr)


result = (1/prod(harr));
for j = 1:numel(train)
    result = result * uni((teste(j) - train(j))/harr(j));
end

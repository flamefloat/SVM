load('C:\Users\MH\Desktop\MyCode\MLalgorithm\SVM\Data1.mat','data')
for i=1:length(data)
    data(i,1) = roundn(data(i,1),-3);
    data(i,2) = roundn(data(i,2),-3);
end
c=[];
for i=1:length(data)
    if data(i,3)==1
        c(i,:)=[0,255,255];
    else
        c(i,:)=[0,0,255];
    end
end
scatter(data(:,1),data(:,2),[],c)
save('C:\Users\MH\Desktop\MyCode\MLalgorithm\SVM\Data3.txt','data')
clc;clear;close all
MI_path = 'F:\Python\untitled\PTB_tesr\病人间\数据\MI_PTB';
health_path = 'F:\Python\untitled\PTB_tesr\病人间\数据\health_PTB';

load([MI_path,'\m_train.mat'])
%load([MI_path,'\m_test.mat'])
load([health_path,'\h_train.mat'])
%load([health_path,'\h_test.mat'])
m_train=mapminmax(m_train,0,1);
h_train=mapminmax(h_train,0,1);
%m_test=mapminmax(m_test,0,1);
%h_test=mapminmax(h_test,0,1);
m_train(:,7201)=0;
%m_test(:,4801)=0;
h_train(:,7201)=1;

%m_train = m_train(1:20000,:);
%h_test(:,4801)=1;
len_MI_temp = length(m_train);
len_MI = fix(len_MI_temp*0.8);

len_health_temp = length(h_train);
len_health = fix(len_health_temp*0.8);

ptb_train = [m_train(1:len_MI,:);h_train(1:len_health,:)];
ptb_test = [m_train(len_MI:len_MI_temp,:);h_train(len_health:len_health_temp,:)];
clear h_train
clear m_train



ptb_train =ptb_train(randperm(size(ptb_train,1)),:);
ptb_test =ptb_test(randperm(size(ptb_test,1)),:);


save(['F:\Python\untitled\PTB_tesr\病人间\数据\ptb_test'],'ptb_test')
save(['F:\Python\untitled\PTB_tesr\病人间\数据\ptb_train'],'ptb_train')



temp = 0;
for i = 1:3
    for j = 1:4
        
        plot(ptb_test(10, (temp*600+1): (temp+1)*600));
        
        subplot(3,4,temp+1);
        temp = temp + 1;
        
    end
end




x = [92, 100, 108];
y = [100, 100, 100];
stem(x, y, 'filled');
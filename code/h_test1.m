clc;clear;close all
%%%%%%%%%%%%%
% 初始化
path='F:\Python\untitled\PTB_tesr\病人间\数据';
data_de_path=[path,'\health_two\'];
fileFolder=fullfile(data_de_path);
dirOutput=dir(fullfile(fileFolder));
fileNames_data={dirOutput.name}';%文件名cell

n_data=size(fileNames_data,1);
new_folder=[path,'\health_PTB'];
mkdir(new_folder);

len_f=300;   %前取多少个点
len_b=300;   %后取多少个点
len_A=len_f+len_b+1;    %序列总长
beat_all=[];
temp = 105;

for i=3:n_data    % 遍历所有目录    3:20    3:100    
    new_file=[data_de_path,fileNames_data{i,1},'\']
    file2=fullfile(new_file);
    dir2=dir(fullfile(file2,'*.mat'));
    fileNames_data2={dir2.name}';%文件名cell
    fir_files = length(fileNames_data2);
for j=1:fir_files    %遍历目录下的子文件
    %i=1
    beat_one=[];  % 保存一个导联的信号      
    beat12=[]; % 保存12个到导联的信号
    load([new_file,fileNames_data2{j,1}]);       %load 出所有的数据
    len_x_all = [];
   
    len_one = length(ECG);
    [x,y]=liu_R(ECG(1,10:(len_one-10)));  %检测R波  
    %% 截点
    len_x = length(x);
    %% 重新检测R波 准备截点
    for l=1:12
        % ecg_N=['ECG',num2str(l)];   %合并处理,字符串 拼出 ECGn变量来
        % ecg_n=eval(ecg_N);         %把字符串 转化成变量
        
        beat_one= [];
        for q=10:len_x-10

            beat_one = [beat_one;ECG(l,(x(q)-len_f):(x(q)+len_b-1))];
        end
        beat12 = [beat12,beat_one];
        beat_one = [];   
    end
    beat_all=[beat_all;beat12];
    

  


end
disp(['弄完了第' num2str(i-2) '个目录'])
end
h_train = beat_all;
disp('都弄完了')
save([new_folder,'\','h_train'],'h_train')



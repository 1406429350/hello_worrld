clc;clear;close all
%%%%%%%%%%%%%
% ��ʼ��
path='F:\Python\untitled\PTB_tesr\���˼�\����';
data_de_path=[path,'\health_two\'];
fileFolder=fullfile(data_de_path);
dirOutput=dir(fullfile(fileFolder));
fileNames_data={dirOutput.name}';%�ļ���cell

n_data=size(fileNames_data,1);
new_folder=[path,'\health_PTB'];
mkdir(new_folder);

len_f=300;   %ǰȡ���ٸ���
len_b=300;   %��ȡ���ٸ���
len_A=len_f+len_b+1;    %�����ܳ�
beat_all=[];
temp = 105;

for i=3:n_data    % ��������Ŀ¼    3:20    3:100    
    new_file=[data_de_path,fileNames_data{i,1},'\']
    file2=fullfile(new_file);
    dir2=dir(fullfile(file2,'*.mat'));
    fileNames_data2={dir2.name}';%�ļ���cell
    fir_files = length(fileNames_data2);
for j=1:fir_files    %����Ŀ¼�µ����ļ�
    %i=1
    beat_one=[];  % ����һ���������ź�      
    beat12=[]; % ����12�����������ź�
    load([new_file,fileNames_data2{j,1}]);       %load �����е�����
    len_x_all = [];
   
    len_one = length(ECG);
    [x,y]=liu_R(ECG(1,10:(len_one-10)));  %���R��  
    %% �ص�
    len_x = length(x);
    %% ���¼��R�� ׼���ص�
    for l=1:12
        % ecg_N=['ECG',num2str(l)];   %�ϲ�����,�ַ��� ƴ�� ECGn������
        % ecg_n=eval(ecg_N);         %���ַ��� ת���ɱ���
        
        beat_one= [];
        for q=10:len_x-10

            beat_one = [beat_one;ECG(l,(x(q)-len_f):(x(q)+len_b-1))];
        end
        beat12 = [beat12,beat_one];
        beat_one = [];   
    end
    beat_all=[beat_all;beat12];
    

  


end
disp(['Ū���˵�' num2str(i-2) '��Ŀ¼'])
end
h_train = beat_all;
disp('��Ū����')
save([new_folder,'\','h_train'],'h_train')



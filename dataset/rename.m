clc;
i=0;
for lattice_number=[2:6,8]
    i=i+1;
%fig1=figure;
s0=lattice_number;
filename = ['in1_test_inverse', num2str(lattice_number),'.csv'];
filename_new=['input_test_inverse', num2str(i),'.csv'];
movefile(filename,filename_new);

end
return
i=0;
for lattice_number=[2:6,8]
    i=i+1;
%fig1=figure;
s0=lattice_number;
filename = ['in1_training_', num2str(lattice_number),'.csv'];
filename_new=['input_training_', num2str(i),'.csv'];
movefile(filename,filename_new);

end

i=0;
for lattice_number=[2:6,8]
    i=i+1;
%fig1=figure;
s0=lattice_number;
filename = ['Out1_test_', num2str(lattice_number),'.csv'];
filename_new=['Output_test_', num2str(i),'.csv'];
movefile(filename,filename_new);

end


i=0;
for lattice_number=[2:6,8]
    i=i+1;
%fig1=figure;
s0=lattice_number;
filename = ['Out1_training_', num2str(lattice_number),'.csv'];
filename_new=['Output_training_', num2str(i),'.csv'];
movefile(filename,filename_new);
end



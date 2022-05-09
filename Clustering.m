%% Load the data file
%Clustering
load('DataCleaning.mat');
c_data = dataCleaned;
%% convert all categorical data into numeric data

c_data.race = double(c_data.race);
c_data.gender = double(c_data.gender);
c_data.age = double(c_data.age);
c_data.admission_type_id = double(c_data.admission_type_id);
c_data.discharge_disposition_id = double(c_data.discharge_disposition_id);
c_data.admission_source_id = double(c_data.admission_source_id);
c_data.time_in_hospital = double(c_data.time_in_hospital);
c_data.num_lab_procedures = double(c_data.num_lab_procedures);
c_data.num_procedures = double(c_data.num_procedures);
c_data.num_medications = double(c_data.num_medications);
c_data.diag_1 = double(c_data.diag_1);
c_data.diag_2 = double(c_data.diag_2);
c_data.diag_3 = double(c_data.diag_3);
c_data.number_diagnoses = double(c_data.number_diagnoses);
c_data.max_glu_serum = double(c_data.max_glu_serum);
c_data.A1Cresult = double(c_data.A1Cresult);
c_data.metformin = double(c_data.metformin);
c_data.repaglinide = double(c_data.repaglinide);
c_data.nateglinide = double(c_data.nateglinide);
c_data.glimepiride = double(c_data.glimepiride);
c_data.glipizide = double(c_data.glipizide);
c_data.glyburide = double(c_data.glyburide);
c_data.pioglitazone = double(c_data.pioglitazone);
c_data.rosiglitazone = double(c_data.rosiglitazone);
c_data.insulin = double(c_data.insulin);
c_data.change = double(c_data.change);
c_data.diabetesMed = double(c_data.diabetesMed);
c_data.readmitted = double(c_data.readmitted);
c_input = table2array(c_data(:,1:27));

%transpose the input table
c_input = c_input';

%% Creating, training the network

%Creating the network
net = selforgmap([10 10]);
[net,tr] = train(net,c_input);

%train the network
y = net(c_input);

%View the network
view(net);

%% Plots
figure, plotsomtop(net);



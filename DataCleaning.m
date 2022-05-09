%% PSEUDOCODES
% 1. Extract unprocessed data from UCI database
% 2. Preprocess data by removing unwanted and insignificant attributes,
%    removing duplicates and filling in missing values.
% 3. Prepare categorical data as input and setting readmission as target variables
% 4. Classification of cleaned data set using ANN
% 5. Classify data set to predict readmission using SVM algorithm
% 6. Classify data set to predict readmission using random forest algorithm
% 7. Compute confusion matrix for both classifiers
% 8. Plot ROC and AUROC for both classifiers
% 9. Decide the best classifier for our data set
% 10.Perform clustering on the data set


%% DATA PREPROCESSING
% Diabetes dataset imported and stored in variable 'diabetes'
diabetes = readtable("diabetic_data.csv",'TreatAsEmpty',{'?'}); 
diabetes = standardizeMissing(diabetes,'?');

% Remove insignificant attributes,variables with high missing percentage,
% and zero-variance,and store into variable 'data'
data = removevars(diabetes,[1,6,11,12,16,17,18,40,41]); % data now has 101766 samples with 41 attributes

% Remove duplicates of occurences(patient_nbr), then remove patient ID
[Patients,ia] = unique(data.patient_nbr);   
data = data(ia,:);
data = removevars(data,[1]);  % there is now 71518 samples with 40 attributes

% Preparing categorical variables
data.race = categorical(data.race);
data.gender = categorical(data.gender);
data.age  = categorical(data.age);
data.admission_type_id = categorical(data.admission_type_id);
data.discharge_disposition_id = categorical(data.discharge_disposition_id) ;
data.admission_source_id = categorical(data.admission_source_id);
data.time_in_hospital = categorical(data.time_in_hospital);
data.num_lab_procedures = categorical(data.num_lab_procedures);
data.num_procedures = categorical(data.num_procedures);
data.num_medications = categorical(data.num_medications);
data.number_diagnoses = categorical(data.number_diagnoses);
data.max_glu_serum = categorical(data.max_glu_serum);
data.A1Cresult = categorical(data.A1Cresult);
data.metformin = categorical(data.metformin);
data.repaglinide = categorical(data.repaglinide);
data.nateglinide = categorical(data.nateglinide);
data.chlorpropamide = categorical(data.chlorpropamide);
data.glimepiride = categorical(data.glimepiride);
data.acetohexamide = categorical(data.acetohexamide);
data.glipizide = categorical(data.glipizide);
data.glyburide = categorical(data.glyburide);
data.tolbutamide = categorical(data.tolbutamide);
data.pioglitazone = categorical(data.pioglitazone);
data.rosiglitazone = categorical(data.rosiglitazone);
data.acarbose = categorical(data.acarbose);
data.miglitol = categorical(data.miglitol);
data.troglitazone = categorical(data.troglitazone);
data.tolazamide = categorical(data.tolazamide);
data.insulin = categorical(data.insulin);
data.glyburide_metformin = categorical(data.glyburide_metformin);
data.glipizide_metformin = categorical(data.glipizide_metformin);
data.glimepiride_pioglitazone = categorical(data.glimepiride_pioglitazone);
data.metformin_rosiglitazone = categorical(data.metformin_rosiglitazone);
data.metformin_pioglitazone = categorical(data.metformin_pioglitazone);
data.change = categorical(data.change);
data.diabetesMed = categorical(data.diabetesMed);
data.readmitted = categorical(data.readmitted);

% Viewing attributes with missing values
missingElements = ismissing(data);
numMissing = sum(missingElements,1);
data.Properties.VariableNames(numMissing > 0);

% Replace missing values of race attribute with 'Missing'
% Combine 'Asian', 'Hispanic' and 'Other' into one category
data = fillmissing(data,'constant','Missing','DataVariables',{'race'});
otherrace = {'Asian','Hispanic','Other'};
data.race = mergecats(data.race,otherrace);
data.race = renamecats(data.race,{'1','2','3','4'});

% Renaming categories in gender to 1=Female, 2=Male, 3=Unknown
data.gender = renamecats(data.gender,{'1','2','3'});

% Categorizing age groups into 1= 0-30 ,2= 30-60 and 3= >60
oldage1  = {'[0-10)' , '[10-20)', '[20-30)'};
oldage2  = {'[30-40)', '[40-50)', '[50-60)'};
oldage3  = {'[60-70)','[70-80)',',[80-90)','[90-100)'};
data.age = mergecats(data.age,oldage1);
data.age = mergecats(data.age,oldage2);
data.age = mergecats(data.age,oldage3);
data.age = mergecats(data.age,{'[60-70)','[80-90)'});
data.age = renamecats(data.age,{'1','2','3'});

% Categorizing admission_type_id into 2 categories (1=known and 0=unknown)
known  = {'1','2','3','4','7'};
unknown  = {'5','6','8'};
data.admission_type_id = mergecats(data.admission_type_id,known);
data.admission_type_id = mergecats(data.admission_type_id,unknown);
data.admission_type_id = renamecats(data.admission_type_id,{'1','0'});

% Categorizing admission_source_id into 5 categories (r,t,b,u,o)
referral = {'1','2','3'};
transfer = {'4','5','6','10','18','22','25','26'};
birth    = {'11','12','13','14','23','24'};
other    = {'7','8','19'};
unknown  = {'9','15','17','20','21'};
data.admission_source_id = mergecats(data.admission_source_id,referral);
data.admission_source_id = mergecats(data.admission_source_id,transfer);
data.admission_source_id = mergecats(data.admission_source_id,birth);
data.admission_source_id = mergecats(data.admission_source_id,other);
data.admission_source_id = mergecats(data.admission_source_id,unknown);
data.admission_source_id = renamecats(data.admission_source_id,{'r','t','o','u','b'});

% Categorizing discharge_disposition_id and
% Remove disposition ids associated with ‘Expired’
discharge = {'1','2','3','4','5','6','8','15','16','17','22','23','24','27','28','29','30'};
hospice   = {'13','14'};
expired   = {'11','19','20','21'};
unknown   = {'18','25','26'};
other     = {'7','9','10','12'};
data.discharge_disposition_id = mergecats(data.discharge_disposition_id,discharge);
data.discharge_disposition_id = mergecats(data.discharge_disposition_id,hospice);
data.discharge_disposition_id = mergecats(data.discharge_disposition_id,unknown);
data.discharge_disposition_id = mergecats(data.discharge_disposition_id,other);
data.discharge_disposition_id = mergecats(data.discharge_disposition_id,expired);
data(data.discharge_disposition_id == '11', :) = [];    % there are now 70434 samples with 40 attributes
data.discharge_disposition_id = removecats(data.discharge_disposition_id,{'11'}); %Remove 'expired'
data.discharge_disposition_id = renamecats(data.discharge_disposition_id,{'d','o','h','u'});

% Renaming categories in change (1 = change, 0 = no change)
data.change = renamecats(data.change,{'1','0'});

% Renaming categories in diabetesMed (1 = medications were prescribed, 0 = no medications were prescribed)
data.diabetesMed = renamecats(data.diabetesMed,{'0','1'});

% Categorizing target variable, 0 = discharge, 1 = readdmittance
readmittance  = {'<30','>30'};
data.readmitted = mergecats(data.readmitted,readmittance);
data.readmitted = renamecats(data.readmitted,{'1','0'}); 

% Imputing missing values for diag_1, diag_2, and diag_3
data.diag_1 = categorical(data.diag_1);
data.diag_2 = categorical(data.diag_2);
data.diag_3 = categorical(data.diag_3);
data.diag_1 = double(data.diag_1);
data.diag_2 = double(data.diag_2);
data.diag_3 = double(data.diag_3);
data.diag_1 = imputer(data.diag_1,'mean');         
data.diag_2 = imputer(data.diag_2,'mean'); 
data.diag_3 = imputer(data.diag_3,'mean');

% View data summary to check categories and attributes
% Remove zero-variance and near zero-variance attributes
summary(data);
data = removevars(data,[20,22,25,28,29,30,31,33,34,35,36,37]); %70434 samples with 28 meaningful attributes

% Exporting data into Excel file for categorizing lab procedures,diagnoses and medication
xlswrite("data.xlsx",data);

% Once finished categorizing large categories(up to 800 categories) into fewer levels, reimport data into MATLAB 
data = readtable("data.xlsx");

% Change all attributes that are 'cell array' into categorical data
data.race = categorical(data.race);
data.gender = categorical(data.gender);
data.age  = categorical(data.age);
data.admission_type_id = categorical(data.admission_type_id);
data.discharge_disposition_id = categorical(data.discharge_disposition_id) ;
data.admission_source_id = categorical(data.admission_source_id);
data.time_in_hospital = categorical(data.time_in_hospital);
data.num_lab_procedures = categorical(data.num_lab_procedures);
data.num_procedures = categorical(data.num_procedures);
data.num_medications = categorical(data.num_medications);
data.diag_1 = categorical(data.diag_1);
data.diag_2 = categorical(data.diag_2);
data.diag_3 = categorical(data.diag_3);
data.number_diagnoses = categorical(data.number_diagnoses);
data.max_glu_serum = categorical(data.max_glu_serum);
data.A1Cresult = categorical(data.A1Cresult);
data.metformin = categorical(data.metformin);
data.repaglinide = categorical(data.repaglinide);
data.nateglinide = categorical(data.nateglinide);
data.glimepiride = categorical(data.glimepiride);
data.glipizide = categorical(data.glipizide);
data.glyburide = categorical(data.glyburide);
data.pioglitazone = categorical(data.pioglitazone);
data.rosiglitazone = categorical(data.rosiglitazone);
data.insulin = categorical(data.insulin);
data.change = categorical(data.change);
data.diabetesMed = categorical(data.diabetesMed);
data.readmitted = categorical(data.readmitted);

% Renaming categories
data.num_procedures = renamecats(data.num_procedures,{'0','1'});
data.discharge_disposition_id = renamecats(data.discharge_disposition_id,{'1','2','3','4'});
data.admission_source_id = renamecats(data.admission_source_id,{'1','2','3','4','5'});
data.diag_1 = renamecats(data.diag_1,{'1','2','3','4','6','9','7','8'});
data.diag_2 = renamecats(data.diag_2,{'1','2','3','4','5','6','9','7','8'});
data.diag_3 = renamecats(data.diag_3,{'1','2','3','4','5','6','9','7','8'});
data.max_glu_serum = renamecats(data.max_glu_serum,{'1','2','3','4'});
data.A1Cresult = renamecats(data.A1Cresult,{'1','2','0','3'});

% View summary data again to check all attributes are categorical
% Rename cleaned dataset to 'dataCleaned'
summary(data);
dataCleaned = data; 












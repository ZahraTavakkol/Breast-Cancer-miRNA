
"""miRNA.ipynb
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tarfile
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns


""" ## Functions"""

def entropy(target_col):

    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy
  
def InfoGain(data,split_attribute_name, Y):

    total_entropy = entropy(Y)

    positive = data[data[split_attribute_name] > 0].index
    negative = data[data[split_attribute_name] < 0].index

    New_entropy = (len(positive)/data.shape[0])*entropy(Y[positive]) + (len(negative)/data.shape[0])*entropy(Y[negative])
    #Calculate the information gain
    Information_Gain = total_entropy - New_entropy

    return Information_Gain

def Measure (y_test_vals, y_predict_vals): 
    
    tn, fp, fn, tp = confusion_matrix(y_test_vals,y_predict_vals).ravel()
    specificity = tn/(tn+fp)
    sensitivity = tp/(tp+fn)
    accuracy = (tp + tn)/(tn + fp + fn + tp)
    fpr, tpr, thresholds = metrics.roc_curve(y_test_vals, y_predict_vals)
    AUC = metrics.auc(fpr, tpr)

    return (specificity, sensitivity, accuracy, AUC)

def Cal_Specificity(func, miRNA_list, validation_label, validation_data):
  y_pred = [cross_val_predict(func, validation_data[miRNA_list[i:i + 3]], validation_label, cv=10) for i in range(8)]
  specificity = []
  for i in range(len(y_pred)):
    specificity += [Measure(validation_label,y_pred[i])[0]]
  return specificity

def specificity_plot(specificity_SVM, specificity_RF, title):
  plt.plot([i for i in range(1,9)], specificity_SVM, label = 'SVM')
  plt.plot([i for i in range(1,9)], specificity_RF, label = 'Random Forest')
  plt.ylabel('Specificity')
  plt.legend()
  plt.title(title)

def make_tabel(Classifier, Method, func):
  i = 0
  for a in range(len(func)):
    
    for b in range(len(func[0])):
      specificity, sensitivity, accuracy, AUC = func[a][b]
      
      if i == 5:
        print('{:^10}|{:25}|{:^25}|{:^25}|{:^25}|{:^20}'.format(Classifier, Method[i], accuracy, sensitivity, specificity, AUC))
      else:
        print('{:^10}|{:25}|{:^25}|{:^25}|{:^25}|{:^20}'.format('', Method[i], accuracy, sensitivity, specificity, AUC))
      i += 1


""" ## Loading Data """

sample_sheet = pd.read_csv('gdc_sample_sheet.tsv', sep = '\t')
# Extract data
tf = tarfile.open("gdc_download_20210105_232454.032551.tar.gz")
tf.extractall()
num_samples = sample_sheet.shape[0]

dataframes = [pd.read_csv(f, sep = '\t') for f in tf.getnames()[1:] if f.split('/')[1] != 'annotations.txt']
id = [f.split('/')[1] for f in tf.getnames()[1:] if f.split('/')[1] != 'annotations.txt']
Label = sample_sheet.sort_values(by = 'File ID')
Label.index = range(num_samples)

"""## Make Tables"""

table = pd.DataFrame([dataframes[i]['reads_per_million_miRNA_mapped'] for i in range(num_samples)])
table.index = ['Sample_' + str(i) for i in range(1, num_samples + 1)]
table.columns = dataframes[0]['miRNA_ID']

# Final features and Labels

Y = Label['Sample Type'].replace({'Solid Tissue Normal' : 0, 'Primary Tumor': 1, 'Metastatic' : 1}).astype(int)

# shuffle the DataFrame rows 
table = table.sample(frac = 1, random_state = 14) 
Y = Y.sample(frac = 1, random_state = 14)
Y.index = range(Y.shape[0])

Sample_Info = pd.DataFrame(data = (id, Y)).T
Sample_Info.index = table.index
Sample_Info.columns = ['Sample Original Name', 'Label']

features = np.log2(table.values + 1)
scaler = preprocessing.StandardScaler().fit(features)
features_scaled = scaler.transform(features)

table_scaled = pd.DataFrame(features_scaled, index= table.index, columns= table.columns, dtype= float).fillna(0)

# Zero Value miRNA

non_zero = (table_scaled != 0).any(axis=0)
print("Number of non-Zero valued miRNA: ", non_zero[non_zero == True].count())
non_zero_mirna = non_zero[non_zero == True]

# Table with no zero value

table_scaled_no_zero = table_scaled.T.loc[non_zero].T

# Clinical miRNA

# wetlab_miRNAs = "hsa-mir-10b hsa-let-7d hsa-mir-206 hsa-mir-34a hsa-mir-125b-1 hsa-let-7f-1 hsa-mir-17 hsa-mir-27b hsa-mir-145 hsa-let-7f-2 hsa-mir-335 hsa-mir-126 hsa-mir-21 hsa-mir-206 hsa-mir-373 hsa-mir-101-1 hsa-mir-125a hsa-mir-30a hsa-mir-520c hsa-mir-101-2 hsa-mir-17 hsa-mir-30b hsa-mir-27a hsa-mir-146a hsa-mir-125b-2 hsa-mir-203a hsa-mir-221 hsa-mir-146b hsa-let-7a-2 hsa-mir-203b hsa-mir-222 hsa-mir-205 hsa-let-7a-3 has-mir-213 hsa-mir-200c hsa-let-7c hsa-mir-155 hsa-mir-31"
wetlab_miRNAs = "hsa-mir-10b hsa-let-7d hsa-mir-206 hsa-mir-34a hsa-mir-125b-1 hsa-let-7f-1 hsa-mir-17 hsa-mir-27b hsa-mir-145 hsa-let-7f-2 hsa-mir-335 hsa-mir-126 hsa-mir-21 hsa-mir-373 hsa-mir-101-1 hsa-mir-125a hsa-mir-30a hsa-mir-520c hsa-mir-101-2 hsa-mir-30b hsa-mir-27a hsa-mir-146a hsa-mir-125b-2 hsa-mir-203a hsa-mir-221 hsa-mir-146b hsa-let-7a-2 hsa-mir-203b hsa-mir-222 hsa-mir-205 hsa-let-7a-3 hsa-mir-181a-1 hsa-mir-200c hsa-let-7c hsa-mir-155 hsa-mir-31"
wetlab_miRNAs_list = list(wetlab_miRNAs.split(" "))

# Checking paper's clinically verified miRNA list

# wetlab_miRNA_index = []
# wetlab_miRNA_sorted = []

# for i in range(1881):
#   if dataframes[0]["miRNA_ID"][i] in wetlab_miRNAs_list:
    
#     wetlab_miRNA_index.append(i)
#     wetlab_miRNA_sorted.append(dataframes[0]["miRNA_ID"][i])

# print(len(wetlab_miRNA_index))  

# for miRNA in wetlab_miRNAs_list:
#     if miRNA not in wetlab_miRNA_sorted:
        
#         print(miRNA)


# Filter scaled table with clinical miRNA

table_scaled_no_zero_clinical = table_scaled_no_zero[wetlab_miRNAs_list]
table_scaled_no_zero_clinical.index = range(num_samples)


""" ## Variables """

# number of samples for feature selection
num_selected = 207

# Validation set length
val_length = num_samples - num_selected 


"""# Feature Selection"""

Chi2_reported = ['hsa-mir-10b', 'hsa-let-7c', 'hsa-mir-145', 'hsa-mir-125b-2', 
                'hsa-mir-125b-1', 'hsa-mir-335', 'hsa-mir-126', 'hsa-mir-125a', 
                'hsa-let-7a-2', 'hsa-let-7a-3']

Lasso_reported = ['hsa-let-7a-3', 'hsa-let-7c', 'hsa-let-7d', 'hsa-mir-101-1', 
                  'hsa-mir-10b', 'hsa-mir-125b-2', 'hsa-mir-145', 'hsa-mir-206', 
                  'hsa-mir-27b', 'hsa-mir-335']

IG_reported = ['hsa-mir-10b', 'hsa-let-7c', 'hsa-mir-145', 'hsa-mir-125b-1', 
               'hsa-mir-125b-2', 'hsa-mir-335', 'hsa-mir-126', 'hsa-mir-125a',
               'hsa-let-7a-2', 'hsa-let-7a-3']

Intersect_reported = np.intersect1d(Chi2_reported, IG_reported)
print("Intersection length of IG and CHI2 reported miRNAs: ", len(Intersect_reported))
Intersect_reported = np.intersect1d(Intersect_reported, Lasso_reported)
print("Intersection length of (IG or CHI2) with Lasso reported miRNAs: ", len(Intersect_reported), "\n\n\n")

print('{:*^30} {:^17} {:*^30}'.format('','Feature Selection', ''),"\n")
"""Information Gain"""

feature_names = wetlab_miRNAs_list
item_values = [InfoGain(table_scaled_no_zero_clinical.iloc[:num_selected],feature,Y.iloc[:num_selected]) for feature in feature_names]
IG = pd.DataFrame(data = (feature_names, item_values)).T.sort_values(by=[1], ascending=False)
IG.index = range(len(set(feature_names)))
ig_result = IG[:10][0].values
print('Information Gain:\n ')
print(IG[:10],"\n")

print('Intersection of our result and article is: ', np.intersect1d(IG[:10][0].values, IG_reported).shape[0], "\n\n\n")

""" CHI2"""

F, p_val = chi2(table[wetlab_miRNAs_list][:num_selected], Y.iloc[:num_selected])
chi2 = pd.DataFrame(data = (feature_names, F)).T.sort_values(by=[1], ascending=False)
chi2.index = range(len(set(feature_names)))
chi2_result = chi2[:10][0]
print('CHI2:\n ')
print(chi2[:10],"\n")

print('Intersection of our result and article is: ', np.intersect1d(chi2_result, Chi2_reported).shape[0], "\n\n\n")

"""Lasso"""

lasso_reg = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear',max_iter=10000,class_weight='balanced'))
lasso_reg.fit(table_scaled_no_zero_clinical.iloc[:num_selected], Y.iloc[:num_selected])
lasso_reg = pd.DataFrame(data = (feature_names, abs(lasso_reg.estimator_.coef_[0]))).T.sort_values(by=[1], ascending=False)
lasso_reg.index = range(len(set(feature_names)))
Lasso_result = lasso_reg[:10][0].values
print('Lasso:\n ')
print(lasso_reg[:10],"\n")

print('Intersection of our result and article is: ', np.intersect1d(Lasso_result, Lasso_reported).shape[0], "\n\n\n")

"""Mutual Info"""

Mutual = SelectKBest(score_func = mutual_info_classif).fit(table_scaled_no_zero_clinical.iloc[:num_selected], Y.iloc[:num_selected])
Mutual_Info = pd.DataFrame(data = (feature_names, Mutual.scores_)).T.sort_values(by = [1], ascending=False)
Mutual_Info.index = range(len(set(feature_names)))
Mutual_Info_result = Mutual_Info[:10][0].values
print('Mutual Info:\n ')
print(Mutual_Info[:10],"\n")

print('Intersection of our result and article is: ', np.intersect1d(Mutual_Info_result, Chi2_reported).shape[0], "\n\n\n")

"""Rigde"""

Ridge_reg = SelectFromModel(LogisticRegression(penalty='l2', max_iter= 10000,class_weight='balanced'))
Ridge_reg.fit(table_scaled_no_zero_clinical.iloc[:num_selected], Y.iloc[:num_selected])
Ridge_reg = pd.DataFrame(data = (feature_names, abs(Ridge_reg.estimator_.coef_[0]))).T.sort_values(by=[1], ascending=False)
Ridge_reg.index = range(len(set(feature_names)))
Ridge_result = Ridge_reg[:10][0].values
print('Ridge:\n ')
print(Ridge_reg[:10],"\n")

print('Intersection of our result and article is: ', np.intersect1d(Ridge_result, Chi2_reported).shape[0],"\n\n\n")

CHI2_corr = table[chi2[:15][0]].corr()
sns.heatmap(CHI2_corr)
"""# Training"""

validation_data = table_scaled_no_zero_clinical.iloc[num_selected:]
validation_data.index = range(val_length)
validation_label = Y.iloc[num_selected:]
validation_label.index = range(val_length)

svm_linear = SVC(kernel='linear',random_state=0)
svm = SVC(kernel='rbf', random_state=0)
clf = RandomForestClassifier(random_state=0,class_weight='balanced')

# SVM for our Results
specificity_SVM_IG = Cal_Specificity(svm, ig_result, validation_label, validation_data)
specificity_SVM_chi2 = Cal_Specificity(svm, chi2_result, validation_label, validation_data)
specificity_SVM_Lasso = Cal_Specificity(svm, Lasso_result, validation_label, validation_data)

specificity_SVM_Mutual_Info = Cal_Specificity(svm, Mutual_Info_result, validation_label, validation_data)
specificity_SVM_Ridge = Cal_Specificity(svm, Ridge_result, validation_label, validation_data)

# RF for our Results
specificity_RF_IG = Cal_Specificity(clf, ig_result, validation_label, validation_data)
specificity_RF_chi2 = Cal_Specificity(clf, chi2_result, validation_label, validation_data)
specificity_RF_Lasso = Cal_Specificity(clf, Lasso_result, validation_label, validation_data)

specificity_RF_Mutual_Info = Cal_Specificity(clf, Mutual_Info_result, validation_label, validation_data)
specificity_RF_Ridge = Cal_Specificity(clf, Ridge_result, validation_label, validation_data)

"""# Table 2"""

validation_data_no_zero = table_scaled_no_zero.iloc[num_selected:]
validation_data_no_zero.index = range(validation_data_no_zero.shape[0])

non_clinical_no_zero = table_scaled_no_zero.drop(wetlab_miRNAs_list, axis = 1)
validation_data_non_clinical_no_zero = non_clinical_no_zero.iloc[num_selected:]
validation_data_non_clinical_no_zero.index = range(validation_data_non_clinical_no_zero.shape[0])

## SVM-RBF

IG_RBF = [cross_val_predict(svm, validation_data[ig_result[:i]], validation_label, cv=10) for i in [10,5,3]]
IG_RBF = [Measure(n, validation_label) for n in IG_RBF]

CHI2_RBF = [cross_val_predict(svm, validation_data[chi2_result[:i]], validation_label, cv=10) for i in [10,5,3]]
CHI2_RBF = [Measure(n, validation_label) for n in CHI2_RBF]

Lasso_RBF = [cross_val_predict(svm, validation_data[Lasso_result[:i]], validation_label, cv=10) for i in [10,5,3]]
Lasso_RBF = [Measure(n, validation_label) for n in Lasso_RBF]

#All
All_clinical_RBF = cross_val_predict(svm, validation_data[wetlab_miRNAs_list], validation_label, cv=10)
All_clinical_RBF = Measure(All_clinical_RBF, validation_label)

Non_zero_RBF = cross_val_predict(svm, validation_data_no_zero, validation_label, cv=10)
Non_zero_RBF = Measure(Non_zero_RBF, validation_label)

Non_clinical_no_zero_RBF = cross_val_predict(svm, validation_data_non_clinical_no_zero, validation_label, cv=10)
Non_clinical_no_zero_RBF = Measure(Non_clinical_no_zero_RBF, validation_label)

All_RBF = [ Non_zero_RBF,Non_clinical_no_zero_RBF, All_clinical_RBF]
# result
RBF = [All_RBF, IG_RBF, CHI2_RBF, Lasso_RBF]
## RF

IG_RF = [cross_val_predict(clf, validation_data[ig_result[:i]], validation_label, cv=10) for i in [10,5,3]]
IG_RF = [Measure(n, validation_label) for n in IG_RF]

CHI2_RF = [cross_val_predict(clf, validation_data[chi2_result[:i]], validation_label, cv=10) for i in [10,5,3]]
CHI2_RF = [Measure(n, validation_label) for n in CHI2_RF]

Lasso_RF = [cross_val_predict(clf, validation_data[Lasso_result[:i]], validation_label, cv=10) for i in [10,5,3]]
Lasso_RF = [Measure(n, validation_label) for n in Lasso_RF]

#All
All_clinical_RF = cross_val_predict(clf, validation_data[wetlab_miRNAs_list], validation_label, cv=10)
All_clinical_RF = Measure(All_clinical_RF, validation_label)

Non_zero_RF = cross_val_predict(clf, validation_data_no_zero, validation_label, cv=10)
Non_zero_RF = Measure(Non_zero_RF, validation_label)

Non_clinical_no_zero_RF = cross_val_predict(clf, validation_data_non_clinical_no_zero, validation_label, cv=10)
Non_clinical_no_zero_RF = Measure(Non_clinical_no_zero_RF, validation_label)

All_RF = [ Non_zero_RF, Non_clinical_no_zero_RF, All_clinical_RF]
# result
RF = [All_RF, IG_RF, CHI2_RF, Lasso_RF]
## SVM-linear

IG_Linear = [cross_val_predict(svm_linear, validation_data[ig_result[:i]], validation_label, cv=10) for i in [10,5,3]]
IG_Linear = [Measure(n, validation_label) for n in IG_Linear]

CHI2_Linear = [cross_val_predict(svm_linear, validation_data[chi2_result[:i]], validation_label, cv=10) for i in [10,5,3]]
CHI2_Linear = [Measure(n, validation_label) for n in CHI2_Linear]

Lasso_Linear = [cross_val_predict(svm_linear, validation_data[Lasso_result[:i]], validation_label, cv=10) for i in [10,5,3]]
Lasso_Linear = [Measure(n, validation_label) for n in Lasso_Linear]

#All
All_clinical_Linear = cross_val_predict(svm_linear, validation_data[wetlab_miRNAs_list], validation_label, cv=10)
All_clinical_Linear = Measure(All_clinical_Linear, validation_label)

Non_zero_Linear = cross_val_predict(svm_linear, validation_data_no_zero, validation_label, cv=10)
Non_zero_Linear = Measure(Non_zero_Linear, validation_label)

Non_clinical_no_zero_Linear = cross_val_predict(svm_linear, validation_data_non_clinical_no_zero, validation_label, cv=10)
Non_clinical_no_zero_Linear = Measure(Non_clinical_no_zero_Linear, validation_label)

All_Linear = [ Non_zero_Linear, Non_clinical_no_zero_Linear, All_clinical_Linear]
#result
Linear = [All_Linear, IG_Linear, CHI2_Linear, Lasso_Linear]

########
print('{:*^56} {:^20} {:*^56}'.format('','Evaluation (Table 2)', ''),"\n")
print('{:^10} {:^25} {:^25} {:^25} {:^25} {:^20}'.format('Classiﬁer', 'Method', 'Accuracy', 'Sensitivity', 'Speciﬁcity', 'AUC'))
Method = [ 'Non Zero', 'Non Clinical Non Zero', 'All Clinical', 'IG - 10', 'IG - 5', 'IG - 3', 'CHI2 - 10',
          'CHI2 - 5', 'CHI2 - 3', 'Lasso - 10', 'Lasso - 5', 'Lasso - 3']
Classifier = ['RF', 'SVM-RBF', 'SVM']
func = [RF, RBF, Linear]
for j in range(len(Classifier)):
  print('{:-^10} {:-^25} {:-^25} {:-^25} {:-^25} {:-^20}'.format('-', '-', '-', '-', '-', '-'))
  make_tabel(Classifier[j], Method, func[j])

"""# Plot Specificity"""

fig = plt.figure(figsize=(12, 9))
fig.add_subplot(2, 3, 1)
specificity_plot(specificity_SVM_IG, specificity_RF_IG, 'Results - IG')
fig.add_subplot(2, 3, 2)
specificity_plot(specificity_SVM_chi2, specificity_RF_chi2, 'Results - CHI2')
fig.add_subplot(2, 3, 3)
specificity_plot(specificity_SVM_Lasso, specificity_RF_Lasso, 'Results - Lasso')
fig.add_subplot(2, 3, 4)
specificity_plot(specificity_SVM_Mutual_Info, specificity_RF_Mutual_Info, 'Results - Mutual Information')
fig.add_subplot(2, 3, 5)
specificity_plot(specificity_SVM_Ridge, specificity_RF_Ridge, 'Results - Rigde')

plt.show()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from eli5.sklearn import PermutationImportance
from pdpbox import info_plots, get_dataset, pdp, get_dataset, info_plots

import shap
import eli5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('hypertension.csv')
X = df.drop('truhbp',axis=1)
y = df['truhbp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# RF
RF = RandomForestClassifier(max_depth=13, n_estimators=142, min_samples_leaf=2)
result = RF.fit(X_train, y_train)
print(result.score(X_train, y_train))
print(result.score(X_test, y_test))

# RR
reg = linear_model.RidgeClassifier(alpha=.5, fit_intercept ='true', tol=1e-4)
result_Ridge = reg.fit(X_train, y_train)
print(result_Ridge.score(X_train, y_train))
print(result_Ridge.score(X_test, y_test))

# SVM
svm = SVC(C=1, kernel='poly', degree = 2)
result_svm = svm.fit(X_train, y_train)
print(result_svm.score(X_train, y_train))
print(result_svm.score(X_test, y_test))

# MLP
mlp = MLPClassifier(solver='lbfgs', activation='relu', learning_rate_init =0.001, alpha=1e-5,
                    hidden_layer_sizes=(10,46), random_state=9)
result_mlp = mlp.fit(X_train, y_train)
print(result_mlp.score(X_train, y_train))
print(result_mlp.score(X_test, y_test))

# DT
clf = DecisionTreeClassifier(criterion="gini", max_depth=15, )
result_clf = clf.fit(X_train, y_train)
print(result_clf.score(X_train, y_train))
print(result_clf.score(X_test, y_test))

# Adaboost
ada = AdaBoostClassifier(n_estimators=61, learning_rate=1, algorithm='SAMME.R')
ada_result = ada.fit(X_train, y_train)
print(ada_result.score(X_train, y_train))
print(ada_result.score(X_test, y_test))

# Gradient Tree Boost
GBDT = GradientBoostingClassifier(n_estimators=51, learning_rate=0.31, max_depth=1, criterion='friedman_mse')
GBDT_result = GBDT.fit(X_train, y_train)
print(GBDT_result.score(X_train, y_train))
print(GBDT_result.score(X_test, y_test))

# Voting Classifier
clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier(criterion="gini", max_depth=5)
clf3 = GaussianNB()
vot = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('gnb', clf3)], voting='soft', flatten_transform='True')
vot_result = vot.fit(X_train, y_train)
print(vot_result.score(X_train, y_train))
print(vot_result.score(X_test, y_test))

# K-Nearest Neighbors
neigh = KNeighborsClassifier(n_neighbors=16, algorithm='auto', metric='minkowski', leaf_size=30, weights='uniform')
neigh_result = neigh.fit(X_train, y_train)
print(neigh_result.score(X_train, y_train))
print(neigh_result.score(X_test, y_test))

#CM
y_test_pred_RF = RF.predict(X_test)
y_test_pred_proba_RF = RF.predict_proba(X_test)[:,1]
y_train_pred_RF = RF.predict(X_train)
y_train_pred_proba_RF = RF.predict_proba(X_train)[:,1]
y_test_pred_reg = reg.predict(X_test)
y_test_pred_proba_reg = reg.decision_function(X_test)
y_train_pred_reg = reg.predict(X_train)
y_train_pred_proba_reg = reg.decision_function(X_train)
y_test_pred_svm = svm.predict(X_test)
y_test_pred_proba_svm = svm.decision_function(X_test)
y_train_pred_svm = svm.predict(X_train)
y_train_pred_proba_svm = svm.decision_function(X_train)
y_test_pred_mlp = mlp.predict(X_test)
y_test_pred_proba_mlp = mlp.predict_proba(X_test)[:,1]
y_train_pred_mlp = mlp.predict(X_train)
y_train_pred_proba_mlp = mlp.predict_proba(X_train)[:,1]
y_test_pred_clf = clf.predict(X_test)
y_test_pred_proba_clf = clf.predict_proba(X_test)[:,1]
y_train_pred_clf = clf.predict(X_train)
y_train_pred_proba_clf = clf.predict_proba(X_train)[:,1]
y_test_pred_ada = ada.predict(X_test)
y_test_pred_proba_ada = ada.predict_proba(X_test)[:,1]
y_train_pred_ada = ada.predict(X_train)
y_train_pred_proba_ada = ada.predict_proba(X_train)[:,1]
y_test_pred_GBDT = GBDT.predict(X_test)
y_test_pred_proba_GBDT = GBDT.predict_proba(X_test)[:,1]
y_train_pred_GBDT = GBDT.predict(X_train)
y_train_pred_proba_GBDT = GBDT.predict_proba(X_train)[:,1]
y_test_pred_vot = vot.predict(X_test)
y_test_pred_proba_vot = vot.predict_proba(X_test)[:,1]
y_train_pred_vot = vot.predict(X_train)
y_train_pred_proba_vot = vot.predict_proba(X_train)[:,1]
y_test_pred_neigh = neigh.predict(X_test)
y_test_pred_proba_neigh = neigh.predict_proba(X_test)[:,1]
y_train_pred_neigh = neigh.predict(X_train)
y_train_pred_proba_neigh = neigh.predict_proba(X_train)[:,1]
RF_confusion_matrix_model = confusion_matrix(y_test, y_test_pred_RF)
reg_confusion_matrix_model = confusion_matrix(y_test, y_test_pred_reg)
svm_confusion_matrix_model = confusion_matrix(y_test, y_test_pred_svm)
mlp_confusion_matrix_model = confusion_matrix(y_test, y_test_pred_mlp)
clf_confusion_matrix_model = confusion_matrix(y_test, y_test_pred_clf)
ada_confusion_matrix_model = confusion_matrix(y_test, y_test_pred_ada)
GBDT_confusion_matrix_model = confusion_matrix(y_test, y_test_pred_GBDT)
vot_confusion_matrix_model = confusion_matrix(y_test, y_test_pred_vot)
neigh_confusion_matrix_model = confusion_matrix(y_test, y_test_pred_neigh)


# POC
RF_fpr, RF_tpr, RF_thresholds = roc_curve(y_test, y_test_pred_proba_RF)
RFt_fpr, RFt_tpr, RFt_thresholds = roc_curve(y_train, y_train_pred_proba_RF)
reg_fpr, reg_tpr, reg_thresholds = roc_curve(y_test, y_test_pred_proba_reg)
regt_fpr, regt_tpr, regt_thresholds = roc_curve(y_train, y_train_pred_proba_reg)
svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, y_test_pred_proba_svm)
svmt_fpr, svmt_tpr, svmt_thresholds = roc_curve(y_train, y_train_pred_proba_svm)
mlp_fpr, mlp_tpr, mlp_thresholds = roc_curve(y_test, y_test_pred_proba_mlp)
mlpt_fpr, mlpt_tpr, mlpt_thresholds = roc_curve(y_train, y_train_pred_proba_mlp)
clf_fpr, clf_tpr, clf_thresholds = roc_curve(y_test, y_test_pred_proba_clf)
clft_fpr, clft_tpr, clft_thresholds = roc_curve(y_train, y_train_pred_proba_clf)
ada_fpr, ada_tpr, ada_thresholds = roc_curve(y_test, y_test_pred_proba_ada)
adat_fpr, adat_tpr, adat_thresholds = roc_curve(y_train, y_train_pred_proba_ada)
GBDT_fpr, GBDT_tpr, GBDT_thresholds = roc_curve(y_test, y_test_pred_proba_GBDT)
GBDTt_fpr, GBDTt_tpr, GBDTt_thresholds = roc_curve(y_train, y_train_pred_proba_GBDT)
vot_fpr, vot_tpr, vot_thresholds = roc_curve(y_test, y_test_pred_proba_vot)
vott_fpr, vott_tpr, vott_thresholds = roc_curve(y_train, y_train_pred_proba_vot)
neigh_fpr, neigh_tpr, neigh_thresholds = roc_curve(y_test, y_test_pred_proba_neigh)
neight_fpr, neight_tpr, neight_thresholds = roc_curve(y_train, y_train_pred_proba_neigh)

# AUC
RF_auc = auc(RF_fpr, RF_tpr)
RFt_auc = auc(RFt_fpr, RFt_tpr)
# RF_ROC
plt.plot(RF_fpr, RF_tpr, c = '#A1A9D0')
plt.plot(RFt_fpr, RFt_tpr, c = '#F0988C')
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title(' ')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.6,0.15,fontdict=None, s = 'Testing   AUC = %.2f'%RF_auc, c = 'black')
plt.text(0.6,0.08,fontdict=None, s = 'Training  AUC = %.2f'%RFt_auc, c = 'black')
plt.grid(True)

# AUC
reg_auc = auc(reg_fpr, reg_tpr)
regt_auc = auc(regt_fpr, regt_tpr)
# RR_ROC
plt.plot(reg_fpr, reg_tpr, c = '#A1A9D0')
plt.plot(regt_fpr, regt_tpr, c = '#F0988C')
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title(' ')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.6,0.15,fontdict=None, s = 'Testing   AUC = %.2f'%reg_auc, c = 'black')
plt.text(0.6,0.08,fontdict=None, s = 'Training  AUC = %.2f'%regt_auc, c = 'black')
plt.grid(True)

# AUC
svm_auc = auc(svm_fpr, svm_tpr)
svmt_auc = auc(svmt_fpr, svmt_tpr)
# SVM_ROC
plt.plot(svm_fpr, svm_tpr, c = '#A1A9D0')
plt.plot(svmt_fpr, svmt_tpr, c = '#F0988C')
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title(' ')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.6,0.15,fontdict=None, s = 'Testing   AUC = %.2f'%svm_auc, c = 'black')
plt.text(0.6,0.08,fontdict=None, s = 'Training  AUC = %.2f'%svmt_auc, c = 'black')
plt.grid(True)

# AUC
mlp_auc = auc(mlp_fpr, mlp_tpr)
mlpt_auc = auc(mlpt_fpr, mlpt_tpr)
# mlp_ROC
plt.plot(mlp_fpr, mlp_tpr, c = '#A1A9D0')
plt.plot(mlpt_fpr, mlpt_tpr, c = '#F0988C')
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title(' ')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.6,0.15,fontdict=None, s = 'Testing   AUC = %.2f'%mlp_auc, c = 'black')
plt.text(0.6,0.08,fontdict=None, s = 'Training  AUC = %.2f'%mlpt_auc, c = 'black')
plt.grid(True)

# AUC
clf_auc = auc(clf_fpr, clf_tpr)
clft_auc = auc(clft_fpr, clft_tpr)
# clf_ROC
plt.plot(clf_fpr, clf_tpr, c = '#A1A9D0')
plt.plot(clft_fpr, clft_tpr, c = '#F0988C')
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title(' ')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.6,0.15,fontdict=None, s = 'Testing   AUC = %.2f'%clf_auc, c = 'black')
plt.text(0.6,0.08,fontdict=None, s = 'Training  AUC = %.2f'%clft_auc, c = 'black')
plt.grid(True)

# ada
ada_auc = auc(ada_fpr, ada_tpr)
adat_auc = auc(adat_fpr, adat_tpr)
# ada_ROC
plt.plot(ada_fpr, ada_tpr, c = '#A1A9D0')
plt.plot(adat_fpr, adat_tpr, c = '#F0988C')
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title(' ')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.6,0.15,fontdict=None, s = 'Testing   AUC = %.2f'%ada_auc, c = 'black')
plt.text(0.6,0.08,fontdict=None, s = 'Training  AUC = %.2f'%adat_auc, c = 'black')
plt.grid(True)

# AUC
GBDT_auc = auc(GBDT_fpr, GBDT_tpr)
GBDTt_auc = auc(GBDTt_fpr, GBDTt_tpr)
# gbdt_ROC
plt.plot(GBDT_fpr, GBDT_tpr, c = '#A1A9D0')
plt.plot(GBDTt_fpr, GBDTt_tpr, c = '#F0988C')
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title(' ')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.6,0.15,fontdict=None, s = 'Testing   AUC = %.2f'%GBDT_auc, c = 'black')
plt.text(0.6,0.08,fontdict=None, s = 'Training  AUC = %.2f'%GBDTt_auc, c = 'black')
plt.grid(True)

# AUC
vot_auc = auc(vot_fpr, vot_tpr)
vott_auc = auc(vott_fpr, vott_tpr)
# vot_ROC
plt.plot(vot_fpr, vot_tpr, c = '#A1A9D0')
plt.plot(vott_fpr, vott_tpr, c = '#F0988C')
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title(' ')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.6,0.15,fontdict=None, s = 'Testing   AUC = %.2f'%vot_auc, c = 'black')
plt.text(0.6,0.08,fontdict=None, s = 'Training  AUC = %.2f'%vott_auc, c = 'black')
plt.grid(True)

# AUC
neigh_auc = auc(neigh_fpr, neigh_tpr)
neight_auc = auc(neight_fpr,neight_tpr)
# knn_ROC
plt.plot(neigh_fpr, neigh_tpr, c = '#A1A9D0')
plt.plot(neight_fpr, neight_tpr, c = '#F0988C')
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title(' ')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.6,0.15,fontdict=None, s = 'Testing   AUC = %.2f'%neigh_auc, c = 'black')
plt.text(0.6,0.08,fontdict=None, s = 'Training  AUC = %.2f'%neight_auc, c = 'black')
plt.grid(True)


# feature importance analysis
eli5.show_weights(RF,feature_names=feature_names.to_list(), top=35)

# PDP
base_features = df.columns.values.tolist()
base_features.remove('truhbp')
print(base_features)

# Pb
fig, axes, summary_df = info_plots.actual_plot(
    model=RF, X=X_train, feature='blood_Pb', feature_name='Pb',predict_kwds={},
)
plt.savefig('Pb2disease.svg', dpi= 300, format = "svg")
pdp_dist = pdp.pdp_isolate(
    model=RF, dataset=X_test, model_features=base_features,
    feature='log_blood_Pb'
)
fig, axes = pdp.pdp_plot(pdp_dist, 'Pb', center=True, plot_lines=False, frac_to_plot=0.8, plot_pts_dist=True)
plt.savefig('Pb2disease3.svg', dpi= 300, format = "svg")

# Cd
fig, axes, summary_df = info_plots.actual_plot(
    model=RF, X=X_train, feature='log_urine_Cd', feature_name='Cd',predict_kwds={},
)
plt.savefig('Cd2disease.svg', dpi= 300, format = "svg")
pdp_dist = pdp.pdp_isolate(
    model=RF, dataset=X_test, model_features=base_features,
    feature='log_urine_Cd'
)
fig, axes = pdp.pdp_plot(pdp_dist, 'Cd', center=True, plot_lines=False, frac_to_plot=0.8, plot_pts_dist=True)
plt.savefig('Cd2disease3.svg', dpi= 300, format = "svg")

# Tl
fig, axes, summary_df = info_plots.actual_plot(
    model=RF, X=X_train, feature='log_urine_Tl', feature_name='Tl',predict_kwds={},
)
plt.savefig('Tl2disease.svg', dpi= 300, format = "svg")
pdp_dist = pdp.pdp_isolate(
    model=RF, dataset=X_test, model_features=base_features,
    feature='log_urine_Tl'
)
fig, axes = pdp.pdp_plot(pdp_dist, 'Tl', center=True, plot_lines=False, frac_to_plot=0.8, plot_pts_dist=True)
plt.savefig('Tl2disease3.svg', dpi= 300, format = "svg")

# Co
fig, axes, summary_df = info_plots.actual_plot(
    model=RF, X=X_train, feature='log_urine_Co', feature_name='Co',predict_kwds={},
)
plt.savefig('Co2disease.svg', dpi= 300, format = "svg")
pdp_dist = pdp.pdp_isolate(
    model=RF, dataset=X_test, model_features=base_features,
    feature='log_urine_Co'
)
fig, axes = pdp.pdp_plot(pdp_dist, 'Co', center=True, plot_lines=False, frac_to_plot=0.8, plot_pts_dist=True)
plt.savefig('Co2disease3.svg', dpi= 300, format = "svg")

# interact
feat_name1 = 'log_urine_Tl'
nick_name1 = 'Tl'
feat_name2 = 'log_urine_Co'
nick_name2 = 'Co'
inter1 = pdp.pdp_interact(
    model=RF, dataset=X_test, model_features=base_features, features=[feat_name1, feat_name2]
)
fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=inter1, feature_names=[nick_name1, nick_name2]
)
plt.savefig('Tl2Co.svg', dpi= 300, format="svg")

# SHAP
shap.initjs()
explainer = shap.TreeExplainer(RF)
shap_values = explainer.shap_values(X_test)
expected_value = explainer.expected_value

shap.decision_plot(expected_value[1], shap_values[1], X_test, alpha = 0.3, new_base_value=0.5)

idx = 3
patient = X_test.iloc[idx,:]
shap_values_patient = explainer.shap_values(patient)
shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values_patient[1], patient)

print(patient)
print(y[3])

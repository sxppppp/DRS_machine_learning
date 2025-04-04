import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use('Agg')
import xgboost as xgb
from sklearn import model_selection
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                             roc_curve, auc, precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
import time
import warnings

warnings.filterwarnings('ignore')

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(np.__version__))
print('Sklearn: {}'.format(sklearn.__version__))
print('Pandas: {}'.format(pd.__version__))

########################################

feature_names = [
    'id', 'exon_num', 'polyA', 'fpkm', 'as_num',
    'm6A', 'm5C', 'length', 'TE_num', 'CG_content', 'peptide_length','Fickett_score',
    'pI',	'ORF_integrity',	'coding_probability',	'label',
    'A%',	'T%',	'C%',	'G%',	'AA%',	'AT%',	'AC%',	'AG%',
    'TA%',	'TT%',	'TC%',	'TG%',	'CA%',	'CT%',	'CC%',	'CG%',	'GA%',	'GT%',	'GC%',	'GG%',
    'Class',
]

data = pd.read_csv("35_traits.txt",sep='\t',names=feature_names,skiprows=1, usecols=feature_names, na_values=['NA', 'NaN', ''])
data = data.drop('id', axis=1)  # 移除ID列

num_imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(num_imputer.fit_transform(data.drop('Class', axis=1)),
                 columns=feature_names[1:-1])
y = data['Class'].astype(int)


non_scale_cols = ['ORF_integrity', 'label']
scale_cols = [col for col in X.columns if col not in non_scale_cols]

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[scale_cols] = scaler.fit_transform(X_scaled[scale_cols])


##############################################
seed = 1
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X_scaled, y,
    test_size=0.25,
    random_state=seed,
    stratify=y
)

###########################################
models = [
    ("Nearest Neighbors", KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        metric='cosine')
     ),
    ("Random Forest", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        max_features='sqrt',
        min_samples_leaf=5,
        random_state=seed)
     ),
    ("XGBoost", xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.5,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=seed)
     ),
    ("Naive Bayes", GaussianNB()),
    ("SVM Linear", SVC(
        kernel='linear',
        C=0.5,
        probability=True,
        random_state=seed)
     ),
    ("SVM RBF", SVC(
        kernel='rbf',
       C=1.0,
        gamma='scale',
        probability=True,
        random_state=seed)
     )
]

####################################
results_list = []
feature_importance_list = []
output_dir = 'model_results'
os.makedirs(output_dir, exist_ok=True)

roc_data = []
pr_data = []

plt.ioff()
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

for name, model in models:
    print(f"\n{'=' * 40}\n正在处理模型: {name}\n{'=' * 40}")
    start_time = time.time()

    try:
        kfold = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_scores = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)

        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        ########################################
        model_results = {
            'Model': name,
            'CV Accuracy Mean': cv_scores.mean(),
            'CV Accuracy Std': cv_scores.std(),

        ** {f'CV Fold {i + 1}': score for i, score in enumerate(cv_scores)},
        'Test Accuracy': accuracy_score(y_test, y_pred),
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1 Score': report['weighted avg']['f1-score'],
        'Training Time (s)': round(training_time, 2)
        }




        #########################
        importance_data = {'Model': name}
        try:
            if isinstance(model, (RandomForestClassifier)):
                importance = model.feature_importances_

            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])

            else:
                importance = permutation_importance(
                    model, X_test, y_test,
                    n_repeats=5,
                    random_state=seed
                ).importances_mean

            for col, score in zip(X.columns, importance):
                importance_data[col] = score

        except Exception as e:
            print(f"Can't evaluate trait importance: {str(e)}")
            for col in X.columns:
                importance_data[col] = np.nan


        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        results_list.append(model_results)
        feature_importance_list.append(importance_data)

        print(f"\n{name} results:")
        print(pd.DataFrame([model_results]).T)
        print(f"\nConfusion matrix:\n{confusion_matrix(y_test, y_pred)}")

        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_score = model.decision_function(X_test)
                y_prob = (y_score - y_score.min()) / (y_score.max() - y_score.min())

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)

            roc_data.append((fpr, tpr, roc_auc, name))
            pr_data.append((recall, precision, pr_auc, name))
        except Exception as e:
            print(f" {name} curve failed: {str(e)}")

    except Exception as e:
        print(f" {name} failed: {str(e)}")


###############################

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join(output_dir, f"ml_result_{timestamp}.xlsx")

with pd.ExcelWriter(filename, engine='openpyxl') as writer:

    pd.DataFrame(results_list).to_excel(
        writer,
        sheet_name='Model performance',
        index=False,
        float_format="%.4f"
    )

    pd.DataFrame(feature_importance_list).to_excel(
        writer,
        sheet_name='Traits importance',
        index=False,
        float_format="%.4f"
    )

    for sheet_name in writer.sheets:
        worksheet = writer.sheets[sheet_name]
        for col in worksheet.columns:
            max_length = max(len(str(cell.value)) for cell in col)
            worksheet.column_dimensions[col[0].column_letter].width = min(max_length + 2, 25)

print(f"\nResult saved: {filename}")

pdf_path = os.path.join(output_dir, f"model_curves_{timestamp}.pdf")

color_dict = {
    "Random Forest": "#1f77b4",
    "XGBoost": "#ff7f0e",
    "SVM Linear": "#2ca02c",
}
with PdfPages(pdf_path) as pdf:

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    for fpr, tpr, auc_val, name in roc_data:
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_val:.2f})')

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Random Guess')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_aspect('equal', adjustable='box')
    fig.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, pad=20)
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        fontsize=10,
        frameon=False
    )
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    for recall, precision, ap, name in pr_data:
        ax.plot(recall, precision, lw=2, label=f'{name} (AP = {ap:.2f})')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_aspect('equal', adjustable='box')
    fig.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves Comparison', fontsize=14, pad=20)
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        fontsize=10,
        frameon=False
    )
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
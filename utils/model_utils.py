import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, make_scorer, f1_score, classification_report, confusion_matrix
from sklearn import metrics

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from lightgbm import LGBMClassifier

classifiers = {
    'SVC': SVC(probability=True),
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
    'LightGBM': LGBMClassifier(force_col_wise=True)
}

parameters = {
    'SVC': {'kernel':('linear', 'rbf'), 'C':[1, 2, 10]}, # SVC
    'Logistic Regression': {'penalty': ('l1','l2', 'elasticnet'), 'C':[1, 2, 10], 'solver':('bfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'saga')}, # LogisticRegression
    'Random Forest': {'n_estimators': [50, 100, 200], 'criterion': ('gini', 'entropy', 'log_loss')}, # RandomForestClassifier
    'AdaBoost': {'n_estimators': [25, 50, 100,], 'learning_rate':  [0.01, 0.1, 0.5, 1.0]}, # AdaBoostClassifier
    'LightGBM': {'n_estimators':  [50, 100, 200], 'boosting_type': ('gbdt', 'dart', 'rf'), 'learning_rate': [0.01, 0.1, 0.5, 1.0]}, # LightGBM
}

def parameter_model_select(X, y):
    '''
    Define os melhores hiperparâmetros para o modelo usando Grid search selection.

    Parâmetros:
    X : DataFrame
        Conjunto de dados de características.

    y : Series
        Conjunto de dados de rótulos.

    Retorno:
    *Tabela resultados*
    best_params : dict
        Dicionário contendo os melhores parâmetros encontrados para cada modelo.
    '''
    best_params = {}

    scores = {'accuracy' : make_scorer(metrics.accuracy_score),
        'recall'   : make_scorer(metrics.recall_score),
        'precision': make_scorer(metrics.precision_score),
        'f1'       : make_scorer(f1_score)}
    

    model = str(input('Select a model: SVC, Logistic Regression, Random Forest, AdaBoost, LightGBM or all'))
    if model.strip().casefold() in 'all':
        for model_name, model_class in classifiers.items():
            print(f"Searching best parameters for {model_class}...")
            grid = GridSearchCV(estimator=model_class, param_grid=parameters[model_name], scoring=scores)
            grid.fit(X, y)
            best_params[model_name] = {
                    'best_params': grid.best_params_,
                    'best_score': grid.best_score_,
                    'best_estimator': grid.best_estimator_
            }
            print(f"Best parameters for {model}: {grid.best_params_} with score {grid.best_score_}")
            print("-"*50)
            print(pd.DataFrame(grid.cv_results_)[['params',
                                        'mean_test_recall',
                                        'mean_test_precision',
                                        'mean_test_f1']])
            print("-"*50)
    else:
        print(f"Searching best parameters for {model}...")
        grid = GridSearchCV(estimator=classifiers[model], param_grid=parameters[model], scoring=scores, refit=False)
        grid.fit(X, y)
        print(f"Best parameters for {model}: {grid.best_params_} with score {grid.best_score_}")
        print("-"*50)
        print(pd.DataFrame(grid.cv_results_)[['params',
                                'mean_test_accuracy'
                                'mean_test_recall',
                                'mean_test_precision',
                                'mean_test_f1']])
        print("-"*50)



def valid_clf_models(X_train, X_test, y_train, y_test):
    '''
    Realiza a validação do modelo com pontuações, Accuracy, Recall, precision, F1 e Log Loss.

    Parâmetros:
    X_train : DataFrame
        Conjunto de dados de características de treinamento.

    X_test : DataFrame
        Conjunto de dados de características de teste.

    y_train : Series
        Conjunto de dados de rótulos de treinamento.

    y_test : Series
        Conjunto de dados de rótulos de teste.

    retorno:
    *Tabela pontuações*
    '''
    for model_name, model_class  in classifiers.items():
        pipe = Pipeline(steps=[("classifier", model_class)])
        
        pipe.fit(X_train, y_train)
        predict_train = pipe.predict(X_train)
        predict_test = pipe.predict(X_test)

        print(model_class)
        print(' ')
        print(f"F1 Score train: {f1_score(y_train, predict_train):.3f}")
        print(f"Accuracy train: {metrics.accuracy_score(y_train, predict_train):.3f}")
        print(f"Recall train: {metrics.recall_score(y_train, predict_train):.3f}")
        print(f"Precision train: {metrics.precision_score(y_train, predict_train):.3f}")
        print(f"Log loss train: {log_loss(y_train, predict_train):.3f}")
        print('- '*30)
        print(f"F1 Score test: {f1_score(y_test, predict_test):.3f}")
        print(f"Accuracy test: {metrics.accuracy_score(y_test, predict_test):.3f}")
        print(f"Recall test: {metrics.recall_score(y_test, predict_test):.3f}")
        print(f"Precision test: {metrics.precision_score(y_test, predict_test):.3f}")
        print(f"Log loss test: {log_loss(y_test, predict_test):.3f}")
        print('-'*50)

def plot_valid_clf_models(X_train, X_test, y_train, y_test):
    '''
    Plota a pontuação dos modelos com a curva ROC.

    Parâmetros:
    X_train : DataFrame
        Conjunto de dados de características de treinamento.

    X_test : DataFrame
        Conjunto de dados de características de teste.

    y_train : Series
        Conjunto de dados de rótulos de treinamento.

    y_test : Series
        Conjunto de dados de rótulos de teste.

    Retorno:
    *Grafico ROC*
    '''
    for model_name, model_class in classifiers.items():
        pipe = Pipeline(steps=[("classifier", model_class)])
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred)

        auc = round(roc_auc_score(y_test, y_pred), 4) 
        plt.plot(fpr, tpr, label=f'{model_name}, AUC={str(auc)}')
    plt.xlabel('Falso Positivo')
    plt.ylabel('verdadeiro Positivo')
    plt.title('Comparando modelos ROC-CURVE')
    plt.legend(loc="lower right")
    plt.savefig("img/Comparando modelos ROC-CURVE")


def evaluate(y_test, predictions):
    '''
    Avalia o desempenho do modelo.

    Parâmetros:
    y_test : Series
        Conjunto de dados de rótulos de teste.

    predictions : array
        Previsões feitas pelo modelo.
    retorno:
    Matriz de confusão, classification report e F1 score.
    '''
    confusionMatrix = confusion_matrix(y_test, predictions)
    sns.heatmap(confusionMatrix, cmap="Blues", annot=True)
    plt.title("Matriz de confusão")
    print("-"*50)
    print(classification_report(y_test, predictions))
    print("-"*50)
    print(f"F1 Score test: {f1_score(y_test, predictions):.3f}")
       

# preprocessor functions 

def scaler_norm(X):
    '''
    Realiza pré-processamento nos dados com tecnica de normalização.

    Parâmetros:
    X : DataFrame
        Conjunto de dados a ser pré-processado.

    X_column : DataFrame.columns
        Dataframe contendo colunas do conjunto de dados original X.

    Retorno:
    X : array
        Conjunto de dados pré-processado.
    '''
    numerical_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline(
    steps=[("scaler", MinMaxScaler())]
    )

    categorical_transformer = Pipeline(
    steps=[("label", LabelEncoder())]
    )
    preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, numerical_features), ("cat", categorical_transformer, categorical_features)]
    )
    
    X = preprocessor.fit_transform(X)
    return X

def scaler_std(X):
    '''
    Realiza pré-processamento nos dados com tecnica de padronização.

    Parâmetros:
    X : DataFrame
        Conjunto de dados a ser pré-processado.

    X_column : DataFrame.columns
        Dataframe contendo colunas do conjunto de dados original X.

    Retorno:
    X : array
        Conjunto de dados pré-processado.
    '''
    numerical_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("label", LabelEncoder())]
    )

    preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, numerical_features), ("cat", categorical_transformer, categorical_features)]
    )
    X = preprocessor.fit_transform(X)
    return X

def preprocessor_pipeline(X):
    '''
    
    '''
    numerical_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    transformer = ColumnTransformer([
        ('label_encoder', LabelEncoder(), categorical_features),
        ('scaler', StandardScaler(), numerical_features)
    ])

    pipeline = Pipeline([
        ('preprocessamento', transformer),
        ('smote', SMOTE(random_state=42)),
        ('variance_threshold', VarianceThreshold(threshold=0.25)),
        ('scaler', StandardScaler())
    ])
    X = pipeline.fit_transform(X)
    return X
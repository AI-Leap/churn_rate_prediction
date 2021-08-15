# module docstring
'''
This module contains functions for performing feature engineering and
running machine learning algorithms on the churn data to produce
logistic regression and random forest models.
'''

# import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import data_exploration
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    try:
        data_frame = pd.read_csv(pth)
        return data_frame
    except FileNotFoundError:
        print('Error in finding file.')


def perform_eda(data_frame):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''
    data_exploration.draw_histogram(data_frame['Churn'], 'churn_histogram.png')
    data_exploration.draw_histogram(
        data_frame['Customer_Age'],
        'customer_age.png')
    data_exploration.draw_value_counts_barplot(
        data_frame['Marital_Status'], 'marital_status.png')
    data_exploration.draw_distplot(
        data_frame['Total_Trans_Ct'],
        'total_trans_ct.png')
    data_exploration.draw_heatmap(data_frame, 'heat_map.png')


def encode(data_frame, category):
    '''
    helper function to return category column in mean churn value list

    input:
            data_frame: pandas dataframe
            category: the name of the category column

    output:
            mean_churn_list: mean churn value list of the category
    '''
    mean_churn_list = []
    category_groups = data_frame.groupby(category).mean()['Churn']

    for val in data_frame[category]:
        mean_churn_list.append(category_groups.loc[val])

    return mean_churn_list


def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
            [optional argument that could be used for naming variables or index y column]

    output:
            data_frame: pandas dataframe with new columns for
    '''
    for category in category_lst:
        mean_churn_list = encode(data_frame, category)
        data_frame[category + '_Churn'] = mean_churn_list

    return data_frame


def perform_feature_engineering(data_frame, response):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name
              [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    X = pd.DataFrame()
    X[keep_cols] = data_frame[keep_cols]
    y = data_frame[response]

    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return [X_train_1, X_test_1, y_train_1, y_test_1]


def save_roc_image(algorithm1, algorithm2, X, y):
    '''
    function to save roc curve image of two algorithms
    input:
            algorithm1: algorithm 1
            algorithm2: algorithm 2
            X: X training data
            y: y target data
    output:
            None
    '''
    plot1 = plot_roc_curve(algorithm1, X, y)
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    _ = plot_roc_curve(algorithm2, X, y, ax=ax, alpha=0.8)
    plot1.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/roc_image.png')


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/rf_classification_report.png')

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/lr_classification_report.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def save_model(model, model_name):
    '''
    saves model to disk
    input:
            model: model object
            model_name: name of model
    output:
            None
    '''
    joblib.dump(model, './models/' + model_name + '.pkl')


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    save_roc_image(lrc, cv_rfc.best_estimator_, X_test, y_test)

    save_model(lrc, 'logistic_model')
    save_model(cv_rfc.best_estimator_, 'rfc_model')

    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        './images/results/feature_importance.png')


if __name__ == '__main__':
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    data_frame = import_data('./data/bank_data.csv')
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
#     perform_eda(data_frame)
    data_frame = encoder_helper(data_frame, cat_columns, None)

    [X_train, X_test, y_train,
        y_test] = perform_feature_engineering(data_frame, 'Churn')
    train_models(X_train, X_test, y_train, y_test)

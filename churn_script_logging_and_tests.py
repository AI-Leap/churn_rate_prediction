# module docstring
'''
This is a test file to test the functions from churn_library.py
'''

import os
import logging

import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cls.import_data(import_data)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    cls.perform_eda(perform_eda)
    files = os.listdir('./images/eda/')
    try:
        assert len(files) == 5
        assert files == [
            'customer_age.png',
            'marital_status.png',
            'heat_map.png',
            'total_trans_ct.png',
            'churn_histogram.png']
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: There are not required plot images in directory")


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    data_frame = encoder_helper['df']
    cat_columns = encoder_helper['categories']

    encoded_data_frame = cls.encoder_helper(data_frame, cat_columns, None)
    last_five_columns = encoded_data_frame.columns[-5:]
    try:
        assert len(last_five_columns) == len(cat_columns)
        assert sorted(last_five_columns) == sorted(
            [
                'Gender_Churn',
                'Education_Level_Churn',
                'Marital_Status_Churn',
                'Income_Category_Churn',
                'Card_Category_Churn'])
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The number of categorical columns"
                + "is not equal to the number of categorical columns")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    [X_train, X_test, y_train, y_test] = cls.perform_feature_engineering(
        perform_feature_engineering, 'Churn')

    try:
        assert X_train.shape[0] == y_train.shape[0]
        assert X_train.shape[1] == 19
        assert X_test.shape[0] == y_test.shape[0]
        assert X_test.shape[1] == 19
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The number of rows"
                + "in the training and testing data is not equal")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        cls.train_models(
            train_models['X_train'],
            train_models['X_test'],
            train_models['y_train'],
            train_models['y_test'])

        models = os.listdir('./models/')

        assert len(models) == 2
        assert models == ['rfc_model.pkl', 'logistic_model.pkl']
        logging.info("Testing train_models: SUCCESS")
    except Exception as err:
        logging.error(
            "Testing train_models: There was an error while training the models")
        raise err


if __name__ == "__main__":
    test_import('./data/bank_data.csv')

    data_frame = cls.import_data('./data/bank_data.csv')

    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1)
    test_eda(data_frame)

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    test_encoder_helper({
      'df': data_frame,
      'categories': cat_columns
    })

    data_frame = cls.encoder_helper(data_frame, cat_columns, None)
    test_perform_feature_engineering(data_frame)

    [X_train, X_test, y_train,
        y_test] = cls.perform_feature_engineering(data_frame, 'Churn')
    test_train_models({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    })

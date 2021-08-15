# module docstring
'''
This module is used to explore the data and save the figures
'''

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def draw_histogram(data, file_name):
    '''
    Draw histogram for the given data
    input:
          data: pandas series
          file_name: string
    output:
          NONE
    '''
    plt.figure(figsize=(20, 10))
    data.hist()
    plt.savefig('./images/eda/' + file_name)

def draw_distplot(data, file_name):
    '''
    Draw distplot for the given data
    input:
          data: pandas series
          file_name: string
    output:
          NONE
    '''

    plt.figure(figsize=(20, 10))
    sns.distplot(data)
    plt.savefig('./images/eda/' + file_name)

def draw_value_counts_barplot(data, file_name):
    '''
    Draw value counts barplot for the given data
    input:
          data: pandas series
          file_name: string
    output:
          NONE
    '''

    plt.figure(figsize=(20, 10))
    data.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/' + file_name)

def draw_heatmap(data_frame, file_name):
    '''
    Draw heatmap for the given data frame
    input:
          data_frame: pandas data frame
          file_name: string
    output:
          NONE
    '''

    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig('./images/eda/' + file_name)

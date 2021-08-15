import matplotlib.pyplot as plt
import seaborn as sns;
sns.set()

def draw_histogram(data, file_name):
    plt.figure(figsize=(20, 10))
    data.hist();
    plt.savefig('./images/eda/' + file_name)
    
def draw_distplot(data, file_name):
    plt.figure(figsize=(20, 10))
    sns.distplot(data)
    plt.savefig('./images/eda/' + file_name)

def draw_value_counts_barplot(data, file_name):
    plt.figure(figsize=(20, 10))
    data.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/' + file_name)

def draw_heatmap(df, file_name):
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig('./images/eda/' + file_name)
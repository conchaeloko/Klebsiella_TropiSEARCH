"""
Created on 25/09/2023
TropiGAT graph functions
@author: conchaeloko
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def make_table(log_file) :
    lines_data = [line for line in open(log_file).read().split("\n") if line[0:6] == "Epoch:" if len(line.split("\t")) == 6] # if len(line.split("\t")) == 6*
    lines_split = [line.split("\t") for line in lines_data]
    df_raw = pd.DataFrame(lines_split , columns = ["Epoch","Train_loss","Test_loss","MCC","AUC","Acc"])
    df = df_raw.applymap(lambda x: float(x.split(":")[1]))
    df.set_index("Epoch", inplace = True)
    return df
    
def plot_loss(df) : 
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Train_loss'], label='train loss', marker='o', linestyle='-', color ="red")
    plt.plot(df.index, df['Test_loss'], label='test loss', marker='s', linestyle='--', color = "blue")
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)   
    plt.show()  
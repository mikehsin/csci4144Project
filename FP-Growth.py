# importing required module
import pandas as pd
import numpy as np
import plotly.express as px

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

def readInputFile(num):
    # dataset [1] & [2]
    return pd.read_csv("Market_Basket_Optimisation.csv") if num == 1 else pd.read_csv("all_seasons.csv")


# Transform all item of every transactions into the Numpy Array
def getTransaction(transaction, dataset):
    for i in range(0, dataset.shape[0]):
        for j in range(0, dataset.shape[1]):
            transaction.append(dataset.values[i,j])
    return np.array(transaction)

def getPandasDataFrame(transaction):
    return pd.DataFrame(transaction, columns=["items"]) 

#  Getting rid of reduncdent data
def cleaning(fileNum, df):
    if fileNum == 1:
        indexNames = df[df['items'] == "nan" ].index
    else:
        indexNames = df[df['items'] == "None" ].index
    return df.drop(indexNames , inplace=True)

def getTable(df):
    return df.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()

def getViz(df_table, num):
    # set viz title
    df_table["all"] = "Top 50 items" if num == 1 else "Top 50 Universities of producing NBA players"

    # creating tree map using plotly
    return px.treemap(df_table.head(50), path=['all', "items"], values='incident_count',
                    color=df_table["incident_count"].head(50), hover_data=['items'],
                    color_continuous_scale='reds',)

# Create the numpy array of the transactions
def getNumpyArrayTransaction(dataset, transaction):
    transaction = []
    for i in range(dataset.shape[0]):
        transaction.append([str(dataset.values[i,j]) for j in range(dataset.shape[1])])
    return np.array(transaction)

def getTransactionEncoder(transaction):
    transaction_encoder = TransactionEncoder()
    te_ary = transaction_encoder.fit(transaction).transform(transaction)
    return pd.DataFrame(te_ary, columns=transaction_encoder.columns_)


def getTop30(table):
    return table["items"].head(30).values

def FP_Growth(dataset, min_sup):
    return fpgrowth(dataset, min_sup, use_colnames=True)

def main():
    # Transform all item of every transactions into the Numpy Array
    transaction = []
    min_sup = 0.05

    print("######################################################################### ")
    print("Hello, welcome to this demonstration of the FP-Groth algorithm!")
    fileSelection = input("\nPlease enter the number as the choice of file input\n\t(1)Market_Basket_Optimisation.csv \n\t(2)all_seasons.csv\nPlease enter number 1 or 2: ")

    # User input for choosing the file
    while fileSelection != "1" or fileSelection != "2":
        if fileSelection == "1" or fileSelection == "2":
            break
        print("\nInvalid user input\n")
        fileSelection = input("\nPlease enter the number as the choice of file input\n\t(1)Market_Basket_Optimisation.csv \n\t(2)all_seasons.csv\nPlease enter number 1 or 2: ")
        
    dataset = readInputFile(fileSelection)

    # convert the dataset to numpy array
    transaction = getTransaction(transaction, dataset)

    # Transform data to a pandas dataframe and put 1 to every item for makeing countable table 
    df = getPandasDataFrame(transaction)
    df["incident_count"] = 1

    # Cleaning the table
    cleaning(fileSelection, df)

    df_table = getTable(df)

    #  Initial Visualizations
    print(df_table.head(10).style.background_gradient(cmap='Reds'))

    # ploting the treemap
    getViz(df_table, fileSelection).show()

    new_transaction = getNumpyArrayTransaction(dataset, transaction)

    # Initialize the transactionEncoder
    dataset = getTransactionEncoder(new_transaction)
    

    # Get top 30 items and extract them
    top30 = getTop30(df_table)
    dataset = dataset.loc[:,top30]

    # Run the fpgrowth algorithm and print out the result
    print(FP_Growth(dataset, min_sup))

if __name__ == '__main__':
    main()
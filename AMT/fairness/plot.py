import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getColumn(df):
    return list(df.columns.values)

def importData(filename, rows, cols):
    df1 = pd.read_excel (filename, sheet_name='grid-world', nrows=rows, usecols=cols)
    df2 = pd.read_excel (filename, sheet_name='overcooked', nrows=rows, usecols=cols)
    return df1, df2

def reference(df):
    # separate reference answer
    cols_ref = getColumn(df)
    ref_join = df[cols_ref[2:]].to_numpy()
    return ref_join

def matcher(part, case):
    # separate reference answers
    grid_ref = reference(df_ref_grid)
    oc_ref = reference(df_ref_oc)

    if case == 'join_grid':
        boolarr = np.equal(grid_ref[0], part.to_numpy())
    elif case == 'join_oc':
        boolarr = np.equal(oc_ref[0], part.to_numpy())
    return boolarr.sum(axis=0)


def plot_join(join_grid, join_oc, response_n, case):
    plt.figure()
    X = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']
    X_axis = np.arange(len(X))

    if case == 'Grid-world':
        want_legible = np.array([join_grid])
        want_illegible = np.array([response_n-join_grid])
    elif case == 'Overcooked':
        want_legible = np.array([join_oc])
        want_illegible = np.array([response_n-join_oc])

    plt.bar(X_axis - 0.2, want_legible[0], 0.4, label = 'preferred fairness')
    plt.bar(X_axis + 0.2, want_illegible[0], 0.4, label = 'preferred unfairness')
    plt.xticks(X_axis, X)
    plt.ylabel("Number of Participants")
    plt.title(case)
    plt.legend()


# import the data
df_grid, df_oc = importData('AMT.xls', 21, 'A:H')
df_ref_grid, df_ref_oc = importData('Survey_Qs_fairness.xlsx', 2, 'A:J')


def main():
    # compute the total number of participants
    col_sc = getColumn(df_grid)
    response_n = np.size(df_grid[col_sc[0]].to_numpy())


    # count the number of participants willing to join the fair team
    join_grid = matcher(df_grid, 'join_grid')
    join_oc = matcher(df_oc, 'join_oc')


    # plotting predictions
    plot_join(join_grid, join_oc, response_n, 'Grid-world')
    plt.savefig('user_preference_grid.svg')
    plot_join(join_grid, join_oc, response_n, 'Overcooked')
    plt.savefig('user_preference_oc.svg')
    plt.show()



if __name__ == "__main__":
    main()

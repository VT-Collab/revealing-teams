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

def parter(df):
    cols = getColumn(df)
    part_1 = df[cols[:9]]
    part_2 = df[cols[9:18]]
    part_3 = df[cols[18:27]]
    part_4 = df[cols[27:36]]
    join = df[cols[-3:]]
    parts = [part_1, part_2, part_3, part_4, join]
    return parts

def reference(df):
    # separate reference answer
    cols_ref = getColumn(df)
    ref_agent = df[cols_ref[3:6]].to_numpy()
    ref_join = df[cols_ref[6:]].to_numpy()
    return ref_agent, ref_join

def matcher(part, n, case):
    # separate reference answers
    grid_ref_agent, grid_ref_join = reference(df_ref_grid)
    oc_ref_agent, oc_ref_join = reference(df_ref_oc)
    if case == 'pred_grid':
        boolarr = np.equal(grid_ref_agent[n], part.to_numpy())
    elif case == 'join_grid':
        boolarr = np.equal(grid_ref_join[0], part.to_numpy())
    if case == 'pred_oc':
        boolarr = np.equal(oc_ref_agent[n], part.to_numpy())
    elif case == 'join_oc':
        boolarr = np.equal(oc_ref_join[0], part.to_numpy())
    return boolarr.sum()

def Avg(array, case):
    # order part rows: legible, legible, illegible, illegible
    if case == 'oc':
        array[[0,1]] = array[[1,0]]
    array[[0,2]] = array[[2,0]]
    # compute avg score for legible and illegible parts
    avg_lg = np.average(array[0:2], axis=0)
    avg_il = np.average(array[2:], axis=0)
    return np.vstack((avg_lg, avg_il))

def plot_prediction(data, case):
    plt.figure()
    X = ['Scene 1', 'Scene 2', 'Scene 3']
    X_axis = np.arange(len(X))
    plt.bar(X_axis - 0.2, data[0,:], 0.4, label = 'legible')
    plt.bar(X_axis + 0.2, data[1,:], 0.4, label = 'illegible')
    plt.xticks(X_axis, X)
    # plt.xlabel("Scenes")
    plt.ylabel("Number of Correct Predictions")
    if case == 'grid':
        plt.title("Predictions for Grid-world")
    else:
        plt.title("Predictions for Overcooked")
    plt.legend()


def plot_join(join_grid, join_oc, response_n):
    plt.figure()
    X = ['Grid-world', 'Overcooked']
    X_axis = np.arange(len(X))
    want_legible = np.array([join_grid, join_oc])
    want_illegible = np.array([response_n-join_grid,
                            response_n-join_oc])

    plt.bar(X_axis - 0.2, want_legible, 0.4, label = 'legible')
    plt.bar(X_axis + 0.2, want_illegible, 0.4, label = 'illegible')
    plt.xticks(X_axis, X)
    plt.ylabel("Number of Participants")
    plt.legend()


# import the data
df_grid, df_oc = importData('AMT.xlsx', 72, 'A:BZ')
df_ref_grid, df_ref_oc = importData('Survey_Qs_legibility.xlsx', 5, 'A:I')

def main():
    # separate participants' answers
    grid_parts = parter(df_grid)
    oc_parts = parter(df_oc)

    # compute the total number of participants
    col_sc = getColumn(grid_parts[0])
    response_n = np.size(grid_parts[0][col_sc[0:3]].to_numpy())

    preds_grid = np.empty((4,3))
    preds_oc = np.empty((4,3))

    # iterate through different parts of grid-world questions
    for idx, part in enumerate(grid_parts):
        correct_pred = []
        cols = getColumn(part)
        # this loop iterates through scenes 1-3
        for i in range(len(cols)):
            # every three question is a scene
            if idx != 4 and i%3 == 0:
                correct_pred.append(matcher(part[cols[i:i+3]], idx, 'pred_grid'))
            elif idx == 4 and i%3 == 0:
                join_grid = matcher(part[cols[i:i+3]], idx, 'join_grid')
        if idx != 4:
            preds_grid[idx] = correct_pred

    # iterate through different parts of overcoked questions
    for idx, part in enumerate(oc_parts):
        correct_pred = []
        cols = getColumn(part)
        # this loop iterates through scenes 1-3
        for i in range(len(cols)):
            # every three question is a scene
            if idx != 4 and i%3 == 0:
                correct_pred.append(matcher(part[cols[i:i+3]], idx, 'pred_oc'))
            elif idx == 4 and i%3 == 0:
                join_oc = matcher(part[cols[i:i+3]], idx, 'join_oc')
        if idx != 4:
            preds_oc[idx] = correct_pred


    # average predictions over all scenes
    pred_grid = Avg(preds_grid, 'grid')
    pred_oc = Avg(preds_oc, 'oc')


    # plotting predictions
    plot_prediction(pred_grid, 'grid')
    plt.savefig('grid-world.svg')
    plot_prediction(pred_oc, 'oc')
    plt.savefig('overcooked.svg')
    # plot preferences to join teams
    plot_join(join_grid, join_oc, response_n)
    plt.savefig('user_preference.svg')
    plt.show()



if __name__ == "__main__":
    main()

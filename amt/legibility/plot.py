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
    # compute avg and std score for legible and illegible parts
    avg_lg = np.average(array[0:2], axis=0)
    std_lg = np.std(array[0:2], axis=0)

    avg_il = np.average(array[2:], axis=0)
    std_il = np.std(array[2:], axis=0)

    return np.vstack((avg_lg, avg_il)), np.vstack((std_lg, std_il))

def plot_prediction(data_avg, data_std, case):
    plt.figure()
    X = ['Scene 1', 'Scene 2', 'Scene 3']
    X_axis = np.arange(len(X))
    plt.bar(X_axis - 0.2, data_avg[0,:], yerr=data_std[0,:],
            width=0.4, label = 'legible')
    plt.bar(X_axis + 0.2, data_avg[1,:], yerr=data_std[1,:],
            width=0.4, label = 'illegible')
    plt.xticks(X_axis, X)
    plt.ylim([0,100])
    # plt.xlabel("Scenes")
    plt.ylabel("Number of Correct Predictions")
    if case == 'grid':
        plt.title("Predictions for Grid-world")
    else:
        plt.title("Predictions for Overcooked")
    plt.legend()


def plot_join(join_grid, join_oc, participants_n):
    plt.figure()
    print('Grid percentage: ',join_grid/participants_n)
    print('Overcooked percentage: ',join_oc/participants_n)
    X = ['Grid-world', 'Overcooked']
    X_axis = np.arange(len(X))
    want_legible = np.array([join_grid, join_oc])
    want_illegible = np.array([50-join_grid,
                            50-join_oc])

    plt.bar(X_axis - 0.2, want_legible, 0.4, label = 'legible')
    plt.bar(X_axis + 0.2, want_illegible, 0.4, label = 'illegible')
    plt.xticks(X_axis, X)
    plt.ylim([0,50])
    plt.ylabel("Number of Participants")
    plt.legend()


# import the data
df_grid, df_oc = importData('AMT.xlsx', 51, 'A:BZ')
df_ref_grid, df_ref_oc = importData('Survey_Qs_legibility.xlsx', 5, 'A:I')

def main():

    participants_n = 50
    qs_per_scene = 3
    total_n_qs = participants_n*qs_per_scene

    # separate participants' answers
    grid_parts = parter(df_grid)
    oc_parts = parter(df_oc)

    # compute the total number of participants
    col_sc = getColumn(grid_parts[0])

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
                correct_pred.append(matcher(part[cols[i:i+3]], idx, 'pred_grid')/total_n_qs*100)
            elif idx == 4 and i%3 == 0:
                join_grid = matcher(part[cols[i:i+3]], idx, 'join_grid')/3
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
                correct_pred.append(matcher(part[cols[i:i+3]], idx, 'pred_oc')/total_n_qs*100)
            elif idx == 4 and i%3 == 0:
                join_oc = matcher(part[cols[i:i+3]], idx, 'join_oc')/3
        if idx != 4:
            preds_oc[idx] = correct_pred

    # average predictions over all scenes
    avg_pred_grid, std_pred_grid = Avg(preds_grid, 'grid')
    avg_pred_oc, std_pred_oc = Avg(preds_oc, 'oc')

    # plotting predictions
    plot_prediction(avg_pred_grid, std_pred_grid, 'grid')
    plt.savefig('grid-world.svg')
    plot_prediction(avg_pred_oc, std_pred_oc, 'oc')
    plt.savefig('overcooked.svg')
    # plot preferences to join teams
    plot_join(join_grid, join_oc, participants_n)
    plt.savefig('user_preference.svg')
    plt.show()



if __name__ == "__main__":
    main()

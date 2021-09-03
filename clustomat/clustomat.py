import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

class NoReduction:
    def fit_transform(self, vecs):
        return vecs


def cluster(red_model, cluster_model, df):
    vecs = red_model.fit_transform(np.stack(df["answers"]))
    vecs = np.nan_to_num(vecs)
    df["cluster"] = cluster_model.fit_predict(vecs)

    return df


def reduce_2d(reducer_2d, df):
    vecs_2d = reducer_2d.fit_transform(np.stack(df["answers"]))

    df["X"] = vecs_2d[:, 0]
    df["Y"] = vecs_2d[:, 1]

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    df['X'] = scaler.fit_transform(df['X'].values.reshape(-1,1))
    df['Y'] = scaler.fit_transform(df['Y'].values.reshape(-1,1))

    return df


def plot(df, voronoi=False):
    
    if voronoi:
        # make data centers
        centers = []
        for n in df.cluster.unique():
            points = np.array(df.where(df.cluster==n).dropna()[["X","Y"]])
            points = np.nan_to_num(points)
            avgs = np.mean(points, axis=0)
            if n == -1:
                avgs = [avgs, [0.5, 0.5]]
                avgs = np.stack(avgs)
                avgs = np.mean(avgs, axis=0)
            centers.append(avgs)

        points = np.stack(centers)

        # add 4 distant dummy points
        points = np.append(points, [[999,999], [-999,999], [999,-999], [-999,-999]], axis = 0)

        # compute Voronoi tesselation
        vor = Voronoi(points)

        # plot
        plot = voronoi_plot_2d(vor, show_points=False, show_vertices=False)

        size = 12
        plt.figure(figsize=(size,size))

        # colorize
        for region in vor.regions:
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                plt.fill(*zip(*polygon), zorder=0)
    else:
        size = 12
        plt.figure(figsize=(size,size))
    
    # fix the range of axes
    plt.xlim([0,1]), plt.ylim([0,1])

    def_offset = 0.01
    edge = 0.95

    plot = plot = sns.scatterplot(data=df, x="X", y="Y", hue="cluster", palette="tab10", legend=False, zorder=1)

    for line in range(0,df.shape[0]):
        x = df["X"][line]
        y = df["Y"][line]
        len_based_offset = len(df["party"][line]) / 100
        x_text = min(x+def_offset, edge) if x<0.5 else min(x-len_based_offset, edge)
        y_text = min(y+def_offset, edge) if y<0.5 else y-0.025
        plot.text(x_text, y_text, 
                df["party"][line], horizontalalignment='left', 
                size='medium', color='black', weight='semibold')


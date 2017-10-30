import numpy as np
np.random.seed(0)

from bokeh.io import curdoc
from bokeh.layouts import widgetbox, row, column, layout
from bokeh.models import ColumnDataSource, Select, Slider
from bokeh.palettes import Spectral6
from bokeh.io import output_notebook, show
from bokeh.plotting import figure, output_file, show
from sklearn import tree

from sklearn import neighbors, datasets
from sklearn.neighbors import NearestNeighbors
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, MinMaxScaler

x_attribute = [
    'Area',
    'Perimeter', 
    'Compactness', 
    'Length of kernel', 
    'Width of kernel', 
    'Asymmetry coefficient',
    'Length of kernel groove' 
]

y_attribute = [
    'Area',
    'Perimeter', 
    'Compactness', 
    'Length of kernel', 
    'Width of kernel', 
    'Asymmetry coefficient',
    'Length of kernel groove' 
]

metric_choice = [
    'euclidean',
    'manhattan',
    'chebyshev'
]


#assign a color to each label
def get_colors(labels):
    colors=[]
    for i in labels:
        if i==1.0:
            colors.append('red')
        if i==2.0:
            colors.append('blue')
        if i==3.0:
            colors.append('green')
    return colors

data = np.loadtxt('seeds_dataset.txt')
labels= data[:,-1]
features= data[:, range(7)]
scaler=MinMaxScaler() #normalize the data
scaler.fit(features)
features=scaler.transform(features)
TOOLS=["pan, wheel_zoom,box_zoom,box_select,reset"]

actual_data=ColumnDataSource(data=dict(x=features[:,0], y=features[:,1], colors=get_colors(labels)))

knn_plot=figure(plot_width=500, plot_height=500, toolbar_location='right', title='Seed Types: K Nearest Neighbors',tools=TOOLS)
knn_plot.circle('x','y',fill_color='colors', line_color='colors', source=actual_data)

decisiontree_plot=figure(plot_width=500, plot_height=500, x_range=knn_plot.x_range, y_range=knn_plot.y_range, toolbar_location='right', title='Seed Types: Decision Tree',tools=TOOLS)
decisiontree_plot.circle('x','y',fill_color='colors', line_color='colors', source=actual_data)


# find the row associated with each attribute
def get_attribute(attribute):
    if attribute=='Area':
        return features[:,0]
    if attribute=='Perimeter':
        return features[:,1]
    if attribute=='Compactness':
        return features[:,2]
    if attribute=='Length of kernel':
        return features[:,3]
    if attribute=='Width of kernel': 
        return features[:,4]
    if attribute=='Asymmetry coefficient': 
        return features[:,5]        
    if attribute=='Length of kernel groove': 
        return features[:,6]        


# widgets
k_slider = Slider(title="Number of neighbors",
                         value=2.0,
                         start=2.0,
                         end=20.0,
                         step=1,
                         width=400)

depth_slider = Slider(title="Max Depth",
                         value=2.0,
                         start=2.0,
                         end=100.0,
                         step=1,
                         width=400)

split_slider = Slider(title="Min Samples Split",
                         value=2.0,
                         start=2.0,
                         end=100.0,
                         step=1,
                         width=400)

x_attribute_select = Select(value='Area',
                          title='Select X attribute:',
                          width=200,
                          options=x_attribute)

y_attribute_select = Select(value='Perimeter',
                          title='Select Y attribute:',
                          width=200,
                          options=y_attribute)

metric_select = Select(value='euclidean',
                          title='Select metric:',
                          width=200,
                          options=metric_choice)

# classification algorithms 
def clustering(metric, k_neighbors):
    clf = neighbors.KNeighborsClassifier(metric=metric, n_neighbors=k_neighbors)
    return clf

def decisiontree(depth, split):
    clf1=tree.DecisionTreeClassifier(max_depth=depth, min_samples_split=split)
    return clf1

#create underlying decision boundary with many grid of points
boundary_points=[]
for a in np.arange(0,1,0.01):
    for b in np.arange(0,1,0.01):
        boundary_points.append([a,b])
boundary_points=np.array(boundary_points)

clf=clustering('euclidean', 2)
clf.fit(features[:,[0,1]],labels)
predictions=clf.predict(boundary_points)
knn_data=ColumnDataSource(data=dict(x=boundary_points[:,0], y=boundary_points[:,1], colors=get_colors(predictions)))

clf1=decisiontree(2, 2)
clf1.fit(features[:,[0,1]],labels)
predictionsDT=clf1.predict(boundary_points)
decisiontree_data=ColumnDataSource(data=dict(x=boundary_points[:,0], y=boundary_points[:,1], colors=get_colors(predictionsDT)))


knn_plot.square('x','y',line_color='colors', fill_color='colors', size=9.5, alpha=0.05, source=knn_data)
decisiontree_plot.square('x','y',line_color='colors', fill_color='colors', size=9.5, alpha=0.05, source=decisiontree_data)


# set up callbacks
def update_k_neighbors(attrname, old, new):
    k_neighbors = int(k_slider.value)
    selected_x_attribute=get_attribute(x_attribute_select.value)
    selected_y_attribute=get_attribute(y_attribute_select.value)
    selected_metric=metric_select.value

    clf = clustering(selected_metric, k_neighbors)
    clf.fit(np.column_stack((selected_x_attribute, selected_y_attribute)), labels)
    predictions = clf.predict(boundary_points)
    colors = get_colors(predictions)

    knn_data.data = dict(colors=colors, x=boundary_points[:, 0], y=boundary_points[:, 1])

def update_depth(attrname, old, new):
    selected_depth = int(depth_slider.value)
    selected_split = int(split_slider.value)
    selected_x_attribute=get_attribute(x_attribute_select.value)
    selected_y_attribute=get_attribute(y_attribute_select.value)

    clf1 = decisiontree(selected_depth, selected_split)
    clf1.fit(np.column_stack((selected_x_attribute, selected_y_attribute)), labels)
    predictionsDT = clf1.predict(boundary_points)
    colors = get_colors(predictionsDT)

    decisiontree_data.data = dict(colors=colors, x=boundary_points[:, 0], y=boundary_points[:, 1])

def update_split(attrname, old, new):
    selected_depth = int(depth_slider.value)
    selected_split = int(split_slider.value)
    selected_x_attribute=get_attribute(x_attribute_select.value)
    selected_y_attribute=get_attribute(y_attribute_select.value)

    clf1 = decisiontree(selected_depth, selected_split)
    clf1.fit(np.column_stack((selected_x_attribute, selected_y_attribute)), labels)
    predictionsDT = clf1.predict(boundary_points)

    decisiontree_data.data = dict(colors=get_colors(predictionsDT), x=boundary_points[:, 0], y=boundary_points[:, 1])

def update_x_attribute(attrname, old, new):
    selected_x_attribute=get_attribute(x_attribute_select.value)
    selected_y_attribute=get_attribute(y_attribute_select.value)
    knn_plot.xaxis.axis_label = x_attribute_select.value
    decisiontree_plot.xaxis.axis_label = x_attribute_select.value
    selected_metric=metric_select.value

    clf = clustering(selected_metric, int(k_slider.value))
    clf.fit(np.column_stack((selected_x_attribute, selected_y_attribute)), labels)
    predictions = clf.predict(boundary_points)

    clf1 = decisiontree(int(depth_slider.value), int(split_slider.value))
    clf1.fit(np.column_stack((selected_x_attribute, selected_y_attribute)), labels)
    predictionsDT = clf1.predict(boundary_points)

    knn_data.data = dict(colors=get_colors(predictions), x=boundary_points[:,0], y=boundary_points[:, 1])
    actual_data.data = dict(colors=get_colors(labels), x=selected_x_attribute, y=selected_y_attribute)
    decisiontree_data.data = dict(colors=get_colors(predictionsDT), x=boundary_points[:, 0], y=boundary_points[:, 1])

def update_y_attribute(attrname, old, new):
    selected_x_attribute=get_attribute(x_attribute_select.value)
    selected_y_attribute=get_attribute(y_attribute_select.value)
    knn_plot.yaxis.axis_label = y_attribute_select.value
    decisiontree_plot.yaxis.axis_label = y_attribute_select.value
    selected_metric=metric_select.value

    clf = clustering(selected_metric, int(k_slider.value))
    clf.fit(np.column_stack((selected_x_attribute, selected_y_attribute)), labels)
    predictions = clf.predict(boundary_points)

    clf1 = decisiontree(int(depth_slider.value), int(split_slider.value))
    clf1.fit(np.column_stack((selected_x_attribute, selected_y_attribute)), labels)
    predictionsDT = clf1.predict(boundary_points)

    knn_data.data = dict(colors=get_colors(predictions), x=boundary_points[:,0], y=boundary_points[:, 1])
    actual_data.data = dict(colors=get_colors(labels), x=selected_x_attribute, y=selected_y_attribute)
    decisiontree_data.data = dict(colors=get_colors(predictionsDT), x=boundary_points[:, 0], y=boundary_points[:, 1])

def update_metric(attrname, old, new):
    selected_x_attribute=get_attribute(x_attribute_select.value)
    selected_y_attribute=get_attribute(y_attribute_select.value)
    knn_plot.yaxis.axis_label = y_attribute_select.value
    selected_metric=metric_select.value

    clf = clustering(selected_metric, int(k_slider.value))
    clf.fit(np.column_stack((selected_x_attribute, selected_y_attribute)), labels)
    predictions = clf.predict(boundary_points)
    colors=get_colors(predictions)

    knn_data.data = dict(colors=colors, x=boundary_points[:,0], y=boundary_points[:, 1])
    actual_data.data = dict(colors=get_colors(labels), x=selected_x_attribute, y=selected_y_attribute)

k_slider.on_change('value', update_k_neighbors)
depth_slider.on_change('value',update_depth)
split_slider.on_change('value', update_split)
x_attribute_select.on_change('value', update_x_attribute)
y_attribute_select.on_change('value', update_y_attribute)
metric_select.on_change('value',update_metric)

knn_plot.xaxis.axis_label = x_attribute_select.value
knn_plot.yaxis.axis_label = y_attribute_select.value

decisiontree_plot.xaxis.axis_label = x_attribute_select.value
decisiontree_plot.yaxis.axis_label = y_attribute_select.value

inputs = widgetbox(k_slider, x_attribute_select, y_attribute_select, depth_slider, split_slider)
l = layout([[row(column(knn_plot, k_slider, metric_select),
    column(
        decisiontree_plot,
        depth_slider,
        split_slider,
        ))],
    [row(x_attribute_select, y_attribute_select)],
])

curdoc().add_root(l, inputs)
curdoc().title = "Classification Demo"

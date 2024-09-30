
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import pandas as pd
from scipy.stats import zscore
import networkx as nx
import gravis as gv
import numpy as np
from typing import Dict, List
from visualize_training.constants import HEX_VALUES


def visualize_hmm_loss(data, phase_col: str, epoch_col: str, loss_col: str,
                       hover_data: Dict = None, norm_y: bool = False,
                       log_y: bool = False, hex: List = HEX_VALUES):

    """Visualize HMM loss plot against the Epochs

    Args:
         data (pd.DataFrame): input data containg training runs
         phase_col (str): column name containing the HMM phases info
         epoch_col (str): column name containing Epoch info
         loss_col (str): column name for loss metric of the model
                        hover_data (Dict): dictionary with column names as
                        fields and bool
         value against them for visibility in tooltip during hover

    """
    data['color'] = data.apply(lambda x: hex[x['phases']-1 % len(hex)], axis=1)
    x = data[epoch_col]
    x_label = epoch_col
    if log_y:
        y = np.log(data[loss_col])
        y_label = "log_"+loss_col
    elif norm_y:
        y = np.apply_along_axis(zscore, 0, data[loss_col].to_numpy())
        y_label = "norm_"+loss_col
    else:
        y = data[loss_col]
        y_label = loss_col

    fig = px.scatter(x=x, y=y, labels={"x": x_label, "y": y_label},
                     hover_data={"phase": data[phase_col]})

    fig.update_traces(marker=dict(
        color=data['color']))

    fig.add_trace(go.Scatter(mode='lines',
                             x=x, y=y,
                             line_color='black', line_width=1,
                             showlegend=False, line=dict(dash='dot'),
                             hoverinfo='skip'))

    fig.update_xaxes(gridcolor="lightgrey", linecolor="black", mirror=True)
    fig.update_yaxes(gridcolor="lightgrey", linecolor="black", mirror=True)

    fig.update_layout(
        title=f"{x_label} vs {y_label}",
        plot_bgcolor='white'
        )

    fig.show()


def visualize_dag(transmat, node_hover_dict: Dict = None, 
                  edge_hover_dict: Dict = None, hex: List = HEX_VALUES):

    """Visualize HMM State Transitions

    Args:
         transmat (np.array): state transition matrix
         hover_dict (Dict): phase wise list of important features to be shown 
                            in tooltip on hover
         HEX_VALUES (List): List containing HEX values of the color pallette
    Output:
        Interactive Graph Visualization with a toggle menu on the right to
        control graph features
    """

    n = transmat.shape[0]
    dot = nx.DiGraph(directed=True)
    for i in range(n):
        for j in range(n):
            if np.round(transmat[i][j], 2) > 0:
                if node_hover_dict:
                    dot.add_node(i, label=str(i+1), color=hex[i % len(hex)],
                                 size=20, hover=node_hover_dict[str(i+1)],
                                 label_size=10)
                else:
                    dot.add_node(i, label=str(i+1), color=hex[i % len(hex)],
                                 size=20)
                if i!=j:
                    if edge_hover_dict:
                        if edge_hover_dict[str(i)+">>"+str(j)]['feature_changes'] != []:
                            dot.add_edge(i, j, label=np.round(transmat[i][j], 3),
                                         label_size=10, length=10,
                                         hover=edge_hover_dict[str(i)+">>"+str(j)]['cols'])
                    else:
                        dot.add_edge(i, j, label=np.round(transmat[i][j], 3),
                                     label_size=10, length=10)

    return gv.d3(dot, edge_label_data_source='label', show_edge_label=True,
                 node_label_data_source='label', node_hover_neighborhood=True,
                 node_hover_tooltip=True, edge_hover_tooltip=True,
                 use_many_body_force_max_distance=True, node_drag_fix=True,
                 layout_algorithm_active=True)


def visualize_avg_log_likelihood(data, dataset_name,
                                 max_components=8):
    """
    Visualize average log likehood, AIC and BIC values against no of components. 
    This visualization is helpful in choosing the best suited no of components.

    Args:
        data (dict): Dictionary containing all the mean scores, AIC and BIC values
        dataset_name (str): Title of Chart (When working with multiple datasets, naming it after the dataset_name is the recommended approach)
        max_components (int, optional): Max no of components for which HMM model needs to be trained for. Defaults to 8.
    """

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = np.arange(1, max_components + 1)

    fig.add_trace(go.Scatter(
        x=x,
        y=pd.Series(data['mean_scores']) + data['scores_stdev'],
        mode='lines', marker=dict(color="#444"),
        line=dict(width=0), name="log density (upper bound)",
        hoverinfo='skip', showlegend=False))

    fig.add_trace(go.Scatter(
        x=x,
        y=pd.Series(data['mean_scores']) - data['scores_stdev'],
        mode='lines', marker=dict(color="#444"),
        line=dict(width=0), name="log density (lower bound)",
        fillcolor='rgba(68, 68, 68, 0.3)', fill='tonexty', showlegend=False,
        hoverinfo='skip'))

    fig.add_trace(
        go.Scatter(x=x, y=data["aics"], name="AIC (right axis)", marker=dict(
            color='red', size=10),
                   hovertemplate="AIC (right axis): %{y:.2f}"),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(x=x, y=data["bics"], name="BIC (right axis)",marker=dict(
            color='green', size=10), hovertemplate="BIC (right axis): %{y:.2f}"),
        secondary_y=True,
    )

    fig.add_trace(go.Scatter(
            x=x,
            y=data["mean_scores"], name="Avg Log Density (left axis)",
            marker=dict(color='blue', size=10), hoverinfo="skip",
            hovertemplate="<extra></extra><b>No of Components: %{x:,.0f}</b><br>" + "Avg Log Density (left axis): %{y:.2f}<br>"))

    fig.update_layout(
        autosize=False,
        width=1300,
        height=800,
        title=dataset_name,
        hovermode="x unified",
        plot_bgcolor='white'
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Number of HMM components",
                     gridcolor="lightgrey",
                     linecolor="black", mirror=True)

    # Set y-axes titles
    fig.update_yaxes(title_text="Average log density", secondary_y=False,
                     gridcolor="lightgrey", linecolor="black")
    fig.update_yaxes(title_text="AIC/BIC", secondary_y=True,
                     gridcolor="lightgrey", linecolor="black")

    fig.show()


def visualize_all_seeds(data_dir, loss_col: str, log_bool: bool = False):
    """
    Visualize HMM loss plots against the Epochs for all the random seeds

    Args:
        data_dir (str): Path to data files.
        loss_col (str): Name of the loss column to be visualized
        log_bool (bool, optional): If log scale is to be considered or not. Defaults to False.
    """
    loss_values = [
        pd.read_csv(file)[loss_col].rename(loss_col+f'_{x}')
        for x, file in enumerate(glob.glob(data_dir + "*.csv"))
        ]
    df = pd.DataFrame(loss_values).T
    df[loss_col + "_avg"] = df.mean(axis=1)

    if log_bool:
        df = df.apply(np.log, axis=1)

    fig = px.line(df, y=df.columns[:-1])
    fig.update_traces(line_color='#FF7F7F', line_width=2)
    fig.update_traces(
        hovertemplate="<br>".join(["%{y:.2f}"]))
    fig.add_trace(go.Scatter(mode='lines', y=df.iloc[:, -1],
                             line_color='black', line_width=4,
                             name=f"{loss_col}_avg", showlegend=True,
                             line=dict(dash='dot')))
    fig.update_traces(
        hovertemplate="<br>".join(["%{y:.2f}"]))
    fig.update_xaxes(gridcolor="lightgrey", linecolor="black", mirror=True)
    fig.update_yaxes(gridcolor="lightgrey", linecolor="black", mirror=True)

    fig.update_layout(
            title=f"all seeds for {loss_col}",
            plot_bgcolor='white',
            yaxis={"title": loss_col}, xaxis={"title": "Epoch"},
            hovermode="x")

    fig.show()


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import gravis as gv
import numpy as np
from typing import Dict, List
from src.constants import HEX_VALUES


def visualize_hmm_loss(data, phase_col: str, epoch_col: str, loss_col: str,
                       hover_data: Dict = None):

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

    fig = px.scatter(data, x=epoch_col, y=loss_col,
                     color=data[phase_col].tolist(), hover_data=hover_data)

    fig.add_trace(go.Scatter(mode='lines',
                             x=data[epoch_col], y=data[loss_col],
                             line_color='black', line_width=1,
                             showlegend=False, line=dict(dash='dot')))
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
    dot = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if np.round(transmat[i][j], 3) > 0:
                if node_hover_dict:
                    dot.add_node(i, label=str(i+1), color=hex[i % len(hex)],
                                 size=10, hover=node_hover_dict[str(i+1)])
                else:
                    dot.add_node(i, label=str(i+1), color=hex[i % len(hex)],
                                 size=10)
                if edge_hover_dict:
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

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = np.arange(1, max_components + 1)

    fig.add_trace(go.Scatter(
            x=x,
            y=data["mean_scores"], error_y=dict(
                type='data',  # value of error bar given in data coordinates
                array=data["scores_stdev"],
                visible=True), name='log density (left axis)'))

    fig.add_trace(
        go.Scatter(x=x, y=data["aics"], name="AIC (right axis)"),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(x=x, y=data["bics"], name="BIC (right axis)"),
        secondary_y=True,
    )

    fig.update_layout(
        autosize=False,
        width=1300,
        height=600,
        title=f"{dataset_name}"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Number of HMM components")

    # Set y-axes titles
    fig.update_yaxes(title_text="Average log density", secondary_y=False)
    fig.update_yaxes(title_text="AIC/BIC", secondary_y=True)

    fig.show()

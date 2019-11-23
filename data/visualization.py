import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

# Use this file to read in your data and prepare the plotly visualizations. The path to the data files are in
# `data/file_name.csv`

def return_figures():
    """Creates four plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """

    # first chart plots arable land from 1990 to 2015 in top 10 economies 
    # as a line chart
    
    uplift_df = pd.read_csv('data/uplift_df_condensed.csv')
    
    cum_gains_df = pd.read_csv('data/cum_gains_df_1221.csv')
    
    clean_df = pd.read_csv('data/clean_data.csv').iloc[:, 2:]
    clean_df = clean_df.loc[clean_df.event == 'offer received']
    
    class_dict = {0: 'Control Non-Responders', 1: 'Control Responders', 2: 'Treatment Non-Responders', 3: 'Treatment Responders'}
    
    clean_df['class_description'] = clean_df.target_class.map(class_dict)
    
    graph_one = []    
    graph_one.append(
      go.Scatter(
      x = cum_gains_df['%population'],
      y = cum_gains_df['%predictors'],
      mode = 'lines',
      name = 'Lift Curve'
      )
    )
    
    graph_one.append(
        go.Scatter(
            x = cum_gains_df['%population'],
            y = cum_gains_df['%population'],
            name = 'Random'
        )
    )

    layout_one = dict(title = 'Cumulative Gains Chart',
                xaxis = dict(title = '% of Population'),
                yaxis = dict(title = '% of Gain'),
                )
    
    # second chart plots    
    graph_two = []
    
    classes = clean_df['class_description'].unique().tolist()

    for class_ in classes:
        trace = go.Histogram(x = clean_df.loc[clean_df['class_description'] == class_].days_as_member.tolist(), opacity=0.75, name = str(class_))
        graph_two.append(trace)

    layout_two = dict(barmode='stack', 
                      xaxis = dict(title='Days As Member'), 
                      yaxis = dict(title='Count'), 
                      title='Count of Events by Member Seniority and Target Class'
                )
    
    # third chart plots percent of population that is rural from 1990 to 2015
    graph_three = []
    for class_ in classes:
        trace = go.Box(x = clean_df.loc[clean_df['class_description'] == class_].days_as_member.tolist(), name = class_)
        graph_three.append(trace)

    layout_three = dict(title = 'Days As Member by Target Class',
                xaxis = dict(title = 'Days as Member')
                       )
    
    # fourth chart shows rural population vs arable land
    channel_melt = pd.melt(clean_df, id_vars = ['id', 'offer_type', 'treatment', 'outcome', 'target_class'], value_vars=['web', 'email', 'mobile', 'social'], var_name="channel")
    
    channel_melt = channel_melt.loc[(channel_melt.value == 1) & (channel_melt.treatment == 1)]
    
    channel_group = pd.DataFrame(channel_melt.groupby(['offer_type', 'channel', 'value']).mean()).reset_index().drop(columns=['value', 'treatment', ]).rename(columns={'outcome': 'conversion_%'})
    
    channel_group['conversion_%'] = channel_group['conversion_%'] * 100
       
    

    graph_four = []
    
    fig_four = px.bar(channel_group, x="offer_type", y="conversion_%", color='channel', barmode='group', height=400)
    
    fig_four.update_layout(title='Average Conversion Rate by Offer Type and Channel')

    layout_four = dict(title = 'Chart Four',
                xaxis = dict(title = 'x-axis label'),
                yaxis = dict(title = 'y-axis label'),
                barmode='group'
                )
    
    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    #figures.append(dict(data=graph_four, layout=layout_four))
    figures.append(fig_four)

    return figures

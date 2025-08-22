from dash import Dash, html, dcc, Input, Output
import pandas as pd
import datetime
from datetime import date
from erddapy import ERDDAP
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import theilslopes
import numpy as np
import json

#initializing
app = Dash(__name__, suppress_callback_exceptions=True) #supress callbacks because not all element IDs are present when page is first loaded
g_server = app.server

#connecting to ERDDAP
server = "https://data.pmel.noaa.gov/pmel/erddap"

#gathering submicron
e_sub = ERDDAP(server=server, protocol="tabledap", response="csv")
search_for = "station_barrow_submicron_chemistry"
end_url = e_sub.get_search_url(search_for=search_for, response="csv")
dataset_ID = pd.read_csv(end_url)["Dataset ID"].iloc[0]
e_sub.dataset_id = dataset_ID
df = e_sub.to_pandas()

#gathering supermicron
e_super = ERDDAP(server=server, protocol="tabledap", response="csv")
search_also = "station_barrow_supermicron_chemistry"
end_url_super = e_super.get_search_url(search_for=search_also, response="csv")
dataset_ID_super = pd.read_csv(end_url_super)["Dataset ID"].iloc[0]
e_super.dataset_id = dataset_ID_super
df2 = e_super.to_pandas()

#data prep and calculations SUBMICRON
df['datetime'] = pd.to_datetime(df['time (UTC)'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['nss_SO4_sub1_conc (micrograms m-3)'] = df['IC_total_SO4_sub1_conc (micrograms m-3)'] - (0.177*df['IC_Na_sub1_conc (micrograms m-3)'])
df['nss_K_sub1_conc (micrograms m-3)'] = df['IC_K_sub1_conc (micrograms m-3)'] - (0.02915*df['IC_Na_sub1_conc (micrograms m-3)'])
df['nss_Mg_sub1_conc (micrograms m-3)'] = df['IC_Mg_sub1_conc (micrograms m-3)'] - (0.088*df['IC_Na_sub1_conc (micrograms m-3)'])
df['nss_Ca_sub1_conc (micrograms m-3)'] = df['IC_Ca_sub1_conc (micrograms m-3)'] - (0.01441*df['IC_Na_sub1_conc (micrograms m-3)'])
df['Dust_conc (micrograms m-3)'] = (2.2*df['XRF_Al_sub1_conc (micrograms m-3)']) + (2.49*df['XRF_Si_sub1_conc (micrograms m-3)']) + (1.63*df['XRF_Ca_sub1_conc (micrograms m-3)']) + (2.42*df['XRF_Fe_sub1_conc (micrograms m-3)']) + (1.94*df['XRF_Ti_sub1_conc (micrograms m-3)'])
df['seasalt_sub1_conc (micrograms m-3)'] = df['IC_Cl_sub1_conc (micrograms m-3)'] + (df['IC_Na_sub1_conc (micrograms m-3)']*1.47)
#defining dropdown names
variables = ['mass_sub1', 'IC_Na_sub1', 'IC_NH4_sub1', 'IC_K_sub1', 'IC_Mg_sub1', 'IC_Ca_sub1', 'IC_MSA_sub1', 'IC_Cl_sub1', 'IC_Br_sub1', 'IC_NO3_sub1', 'IC_total_SO4_sub1', 'IC_Oxalate_sub1', 'nss_SO4_sub1', 'nss_K_sub1',
'nss_Mg_sub1', 'nss_Ca_sub1', 'XRF_Na_sub1', 'XRF_Mg_sub1', 'XRF_Al_sub1', 'XRF_Si_sub1', 'XRF_P_sub1', 'XRF_S_sub1', 'XRF_Cl_sub1', 'XRF_K_sub1', 'XRF_Ca_sub1', 'XRF_Ti_sub1', 'XRF_V_sub1', 
'XRF_Cr_sub1', 'XRF_Mn_sub1', 'XRF_Fe_sub1', 'XRF_Co_sub1', 'XRF_Ni_sub1', 'XRF_Cu_sub1', 'XRF_Zn_sub1', 'XRF_Ga_sub1', 'XRF_Ge_sub1', 'XRF_As_sub1', 'XRF_Se_sub1', 'XRF_Br_sub1', 'XRF_Rb_sub1', 'XRF_Sr_sub1', 'XRF_Y_sub1', 'XRF_Zr_sub1',
'XRF_Mo_sub1', 'XRF_Pd_sub1', 'XRF_Ag_sub1', 'XRF_Cd_sub1', 'XRF_In_sub1', 'XRF_Sn_sub1', 'XRF_Sb_sub1', 'XRF_Ba_sub1', 'XRF_La_sub1', 'XRF_Hg_sub1', 'XRF_Pb_sub1', 'Dust', 'seasalt_sub1']
#list to use later bc nss values have no associated flag columns
no_flag_list = ['nss_SO4_sub1_conc (micrograms m-3)', 'nss_K_sub1_conc (micrograms m-3)', 'nss_Mg_sub1_conc (micrograms m-3)', 'nss_Ca_sub1_conc (micrograms m-3)', 'Dust_conc (micrograms m-3)', 'seasalt_sub1_conc (micrograms m-3)']

#data prep and calculations SUPERMICRON
df2['datetime'] = pd.to_datetime(df2['time (UTC)'])
df2['year'] = df2['datetime'].dt.year
df2['month'] = df2['datetime'].dt.month
df2['nss_SO4_super1_conc (micrograms m-3)'] = df2['IC_total_SO4_super1_conc (micrograms m-3)'] - (0.177*df2['IC_Na_super1_conc (micrograms m-3)'])
df2['nss_K_super1_conc (micrograms m-3)'] = df2['IC_K_super1_conc (micrograms m-3)'] - (0.02915*df2['IC_Na_super1_conc (micrograms m-3)'])
df2['nss_Mg_super1_conc (micrograms m-3)'] = df2['IC_Mg_super1_conc (micrograms m-3)'] - (0.088*df2['IC_Na_super1_conc (micrograms m-3)'])
df2['nss_Ca_super1_conc (micrograms m-3)'] = df2['IC_Ca_super1_conc (micrograms m-3)'] - (0.01441*df2['IC_Na_super1_conc (micrograms m-3)'])
df2['seasalt_super1_conc (micrograms m-3)'] = df2['IC_Cl_super1_conc (micrograms m-3)'] + (df2['IC_Na_super1_conc (micrograms m-3)']*1.47)
#handling data flags and nan values
variables_super = ['IC_Na_super1', 'IC_NH4_super1', 'IC_K_super1', 'IC_Mg_super1', 'IC_Ca_super1', 'IC_MSA_super1', 'IC_Cl_super1', 'IC_Br_super1', 'IC_NO3_super1', 'IC_total_SO4_super1', 'IC_Oxalate_super1', 'nss_SO4_super1', 'nss_K_super1',
'nss_Mg_super1', 'nss_Ca_super1', 'seasalt_super1']
#list to use later bc nss values have no associated flag columns
no_flag_list_super = ['nss_SO4_super1_conc (micrograms m-3)', 'nss_K_super1_conc (micrograms m-3)', 'nss_Mg_super1_conc (micrograms m-3)', 'nss_Ca_super1_conc (micrograms m-3)', 'seasalt_super1_conc (micrograms m-3)']


# Basic app layout
app.layout = html.Div([
    dcc.Store(id='stored-data'),    
    html.Div(
    html.H1('Barrow Observatory Data Display'), # Page title
            style={'textAlign': 'center', 'marginBottom': '30px'}
    ),
    dcc.RadioItems(
        id='dataset-selector',
        options=[
            {'label': 'Submicron', 'value': 'sub'},
            {'label': 'Supermicron', 'value': 'super'}
        ],
        value='sub',
        labelStyle={'display': 'inline-block'}
    ),
    html.Hr(),
    html.Div(id='dynamic-layout')  # This will hold your layout content
])
  
def submicron_layout(): 
    return html.Div([
        html.H3("DATA NOTES:"),
        html.Ul([
            html.Li("Calculations for nonseasalt species are performed based off of conc. of Na with k values of 0.177, 0.02915, 0.088, and 0.01441 for SO4, K, Mg, and Ca respectively (Moffett et al 2020) "),
            html.Li("Seasalt equation used: [seasalt] =  [Cl] + 1.47[Na] (Holland 1978)"),
            html.Li("Dust equation used: [Dust] = 2.2[Al] + 2.49[Si] + 1.63[Ca] + 2.42[Fe] + 1.94[Ti] (Seinfeld 1986, Malm et al. 1994, Perry et al. 1997)")
        ]),
        html.Div(
            html.H2('Plot All Submicron Data'),
            style={'textAlign': 'left', 'marginBottom': '15px'}
        ),
        html.Label('Select a variable to display:'),
        dcc.Dropdown( #first dropdown for graph 1
            id='analyte-dropdown',
            options = [
                {'label': f"{analyte}_conc (micrograms m-3)", 'value': f"{analyte}_conc (micrograms m-3)"}
                for analyte in variables
            ],
            value='mass_sub1_conc (micrograms m-3)',
            clearable=False
        ),
        html.Label('Select a second variable to display:'),
        dcc.Dropdown( #first dropdown for graph 1
            id='analyte-dropdown-2',
            options=[
                {'label': f"{analyte}_conc (micrograms m-3)", 'value': f"{analyte}_conc (micrograms m-3)"}
                for analyte in variables
            ],
            value=None,  # default for second analyte
            clearable=True
        ),
        dcc.Graph(
            id='all-data-graph',
            style={'width': '80%', 'height': '600px', 'marginLeft':'auto', 'marginRight':'auto'}
        ), # graph 1
        html.Div(
            html.H2('Plot Submicron Data Averages'),
            style={'textAlign': 'left', 'marginBottom': '15px'}
        ),
        html.Label('Select a variable to display:'),
        dcc.Dropdown( #first dropdown for graph 2
            id='analyte-dropdown-3',
            options = [
                {'label': f"{analyte}_conc (micrograms m-3)", 'value': f"{analyte}_conc (micrograms m-3)"}
                for analyte in variables
            ],
            value='mass_sub1_conc (micrograms m-3)',
            clearable=False
        ),
        html.Label('Select a second variable to display:'),
        dcc.Dropdown( #second dropdown for graph 2
            id='analyte-dropdown-4',
            options=[
                {'label': f"{analyte}_conc (micrograms m-3)", 'value': f"{analyte}_conc (micrograms m-3)"}
                for analyte in variables
            ],
            value=None,  # default for second analyte
            clearable=True
        ),
        html.Label("Select Mode:"),
        dcc.RadioItems(
            id='mode-selector',
            options=[
                {'label': 'Monthly Average', 'value': 'monthly'},
                {'label': 'Seasonal Average', 'value': 'seasonal'}
            ],
            value='monthly',
            inline=True
        ),
        html.Div([ #using a call back so that this is only visible if seasonal option is selected
            html.Label("Start Month:"),
            dcc.Dropdown(
                id='start-month-dropdown',
                options=[{'label': datetime.date(1900, m, 1).strftime('%B'), 'value': m} for m in range(1, 13)],
                value=1,
                clearable=False,
                style={'width': '150px'}
            ),
            html.Label("End Month:"),
            dcc.Dropdown(
                id='end-month-dropdown',
                options=[{'label': datetime.date(1900, m, 1).strftime('%B'), 'value': m} for m in range(1, 13)],
                value=4,
                clearable=False,
                style={'width': '150px'}
            ),
        ], 
        id='custom-months-container', style={'display': 'none', 'marginBottom': '20px'}
        ),
        html.Label('More Options:'),
        dcc.Checklist( #Option to display standard deviation bars
            id='checklist',
            options=[
                {'label': 'Standard Deviation Bars', 'value': 'std_dev'}
            ],
            value=[]
        ),
        html.Label('Year Range:'),
        dcc.RangeSlider(1997, 2024, 1, 
        value=[1997, 2024], 
        id='my-range-slider',
        marks={
            1997: '1997',
            2000: '2000',
            2005: '2005',
            2010: '2010',
            2015: '2015',
            2020: '2020',
            2024: '2024'
        },
        tooltip={"placement": "bottom", "always_visible": True},
        #allowCross=False,
        ),
        html.Div(id='output-container-range-slider'
        ),
        dcc.Graph( # graph 2
            id='average-graph',
            style={'width': '80%', 'height': '600px', 'marginLeft':'auto', 'marginRight':'auto'}
            ), 
        html.Div(
            html.H2('Submicron Species Ratios over Time Plot'),
            style={'textAlign': 'left', 'marginBottom': '15px'}
        ),
        html.Label('Select a variable to go in the numerator:'),
        dcc.Dropdown( #first dropdown for graph 3
            id='numerator-dropdown',
            options = [
                {'label': f"{analyte}_conc (micrograms m-3)", 'value': f"{analyte}_conc (micrograms m-3)"}
                for analyte in variables
            ],
            value='mass_sub1_conc (micrograms m-3)',
            clearable=False
        ),
        html.Label('Select a variable to go in the denominator:'),
        dcc.Dropdown( #second dropdown for graph 3
            id='denominator-dropdown',
            options = [
                {'label': f"{analyte}_conc (micrograms m-3)", 'value': f"{analyte}_conc (micrograms m-3)"}
                for analyte in variables
            ],
            value='IC_Na_sub1_conc (micrograms m-3)',
            clearable=False
        ),
        dcc.Graph( # graph 3
            id='ratio-graph', 
            style={'width': '80%', 'height': '600px', 'marginLeft':'auto', 'marginRight':'auto'}
            ), 
        ])

def supermicron_layout():
    return html.Div([
        html.H3("DATA NOTES:"),
        html.Ul([
            html.Li("Supermicron mass is not recorded"),
            html.Li("Calculations for nonseasalt species are performed based off of conc. of Na with k values of 0.177, 0.02915, 0.088, and 0.01441 for SO4, K, Mg, and Ca respectively (Moffett et al 2020) "),
            html.Li("Seasalt equation used: [seasalt] =  [Cl] + 1.47[Na] (Holland 1978)"),
        ]),
        html.Div(
            html.H2('Plot All Supermicron Data'),
            style={'textAlign': 'left', 'marginBottom': '15px'}
        ),
        html.Label('Select a variable to display:'),
        dcc.Dropdown( #first dropdown for graph 1
            id='analyte-dropdown',
            options = [
                {'label': f"{analyte}_conc (micrograms m-3)", 'value': f"{analyte}_conc (micrograms m-3)"}
                for analyte in variables_super
            ],
            value='IC_Na_super1_conc (micrograms m-3)',
            clearable=False
        ),
        html.Label('Select a second variable to display:'),
        dcc.Dropdown( #first dropdown for graph 1
            id='analyte-dropdown-2',
            options=[
                {'label': f"{analyte}_conc (micrograms m-3)", 'value': f"{analyte}_conc (micrograms m-3)"}
                for analyte in variables_super
            ],
            value=None,  # default for second analyte
            clearable=True
        ),
        dcc.Graph(
            id='all-data-graph',
            style={'width': '80%', 'height': '600px', 'marginLeft':'auto', 'marginRight':'auto'}
        ), # graph 1
        html.Div(
            html.H2('Plot Supermicron Data Averages'),
            style={'textAlign': 'left', 'marginBottom': '15px'}
        ),
        html.Label('Select a variable to display:'),
        dcc.Dropdown( #first dropdown for graph 2
            id='analyte-dropdown-3',
            options = [
                {'label': f"{analyte}_conc (micrograms m-3)", 'value': f"{analyte}_conc (micrograms m-3)"}
                for analyte in variables_super
            ],
            value='IC_Na_super1_conc (micrograms m-3)',
            clearable=False
        ),
        html.Label('Select a second variable to display:'),
        dcc.Dropdown( #second dropdown for graph 2
            id='analyte-dropdown-4',
            options=[
                {'label': f"{analyte}_conc (micrograms m-3)", 'value': f"{analyte}_conc (micrograms m-3)"}
                for analyte in variables_super
            ],
            value=None,  # default for second analyte
            clearable=True
        ),
        html.Label("Select Mode:"),
        dcc.RadioItems(
            id='mode-selector',
            options=[
                {'label': 'Monthly Average', 'value': 'monthly'},
                {'label': 'Seasonal Average', 'value': 'seasonal'}
            ],
            value='monthly',
            inline=True
        ),
        html.Div([ #only visible if seasonal option is selected
            html.Label("Start Month:"),
            dcc.Dropdown(
                id='start-month-dropdown',
                options=[{'label': datetime.date(1900, m, 1).strftime('%B'), 'value': m} for m in range(1, 13)],
                value=1,
                clearable=False,
                style={'width': '150px'}
            ),
            html.Label("End Month:"),
            dcc.Dropdown(
                id='end-month-dropdown',
                options=[{'label': datetime.date(1900, m, 1).strftime('%B'), 'value': m} for m in range(1, 13)],
                value=4,
                clearable=False,
                style={'width': '150px'}
            ),
        ], 
        id='custom-months-container', style={'display': 'none', 'marginBottom': '20px'}
        ),
        html.Label('More Options:'),
        dcc.Checklist( #Option to display standard deviation bars
            id='checklist',
            options=[
                {'label': 'Standard Deviation Bars', 'value': 'std_dev'}
            ],
            value=[]
        ),
        html.Label('Year Range:'),
        dcc.RangeSlider(1997, 2024, 1, 
        value=[1997, 2024], 
        id='my-range-slider',
        marks={
            1997: '1997',
            2000: '2000',
            2005: '2005',
            2010: '2010',
            2015: '2015',
            2020: '2020',
            2024: '2024'
        },
        tooltip={"placement": "bottom", "always_visible": True},
        #allowCross=False,
        ),
        html.Div(id='output-container-range-slider'
        ),
        dcc.Graph( # graph 2
            id='average-graph',
            style={'width': '80%', 'height': '600px', 'marginLeft':'auto', 'marginRight':'auto'}
            ), 
        html.Div(
            html.H2('Supermicron Species Ratios over Time Plot'),
            style={'textAlign': 'left', 'marginBottom': '15px'}
        ),
        html.Label('Select a variable to go in the numerator:'),
        dcc.Dropdown( #first dropdown for graph 3
            id='numerator-dropdown',
            options = [
                {'label': f"{analyte}_conc (micrograms m-3)", 'value': f"{analyte}_conc (micrograms m-3)"}
                for analyte in variables_super
            ],
            value='IC_Na_super1_conc (micrograms m-3)',
            clearable=False
        ),
        html.Label('Select a variable to go in the denominator:'),
        dcc.Dropdown( #second dropdown for graph 3
            id='denominator-dropdown',
            options = [
                {'label': f"{analyte}_conc (micrograms m-3)", 'value': f"{analyte}_conc (micrograms m-3)"}
                for analyte in variables_super
            ],
            value='IC_NH4_super1_conc (micrograms m-3)',
            clearable=False
        ),
        dcc.Graph( # graph 3
            id='ratio-graph', 
            style={'width': '80%', 'height': '600px', 'marginLeft':'auto', 'marginRight':'auto'}
            ), 
        ])

#callback to change if sub or supermicron data is being stored
@app.callback(
    Output('stored-data', 'data'),
    Input('dataset-selector', 'value')
)
def load_dataset(selected):
    if selected == 'sub':
        return df.to_json(date_format='iso', orient='split')
    elif selected == 'super':
        return df2.to_json(date_format='iso', orient='split')

#callback to change layout based on selection of sub or super
@app.callback(
    Output('dynamic-layout', 'children'),
    Input('dataset-selector', 'value')
)
def switch_layout(selected_dataset):
    if selected_dataset == 'sub':
        return submicron_layout()
    elif selected_dataset == 'super':
        return supermicron_layout()

# Callback for graph #1 (simple timseries)
@app.callback( 
    Output('all-data-graph', 'figure'),
    Input('stored-data', 'data'),
    Input('analyte-dropdown', 'value'),
    Input('analyte-dropdown-2', 'value')
)
def update_graph(data_json, selected_analyte, selected_analyte_2):
    df = pd.read_json(data_json, orient='split')
    fig = go.Figure()
    # adding first variable to graph
    fig.add_trace(go.Scatter(x=df['time (UTC)'], y=df[selected_analyte], mode='markers', name=selected_analyte, yaxis='y1'))
    # add second trace only if selected
    if selected_analyte_2:
        fig.add_trace(go.Scatter(
            x=df['time (UTC)'],
            y=df[selected_analyte_2],
            mode='markers',
            name=selected_analyte_2,
            yaxis='y2'
        ))
        fig.update_layout(
            yaxis2=dict(
                title=selected_analyte_2,
                tickfont=dict(color='red'),
                overlaying='y',
                side='right'
            )
        )
    # Base layout (applies regardless)
    fig.update_layout(
        title="Comparison of Two variables Over Time" if selected_analyte_2 else f"{selected_analyte} Over Time",
        title_x=0.5,
        xaxis=dict(title='Time (UTC)'),
        yaxis=dict(
            title=selected_analyte,
            tickfont=dict(color='blue')
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=60, r=60, t=50, b=50)
    )
    return fig

# Callback for graph #2 (averaging over a period)
@app.callback(
    Output('average-graph', 'figure'),
    Input('stored-data', 'data'),
    Input('dataset-selector', 'value'),
    Input('analyte-dropdown-3', 'value'),
    Input('analyte-dropdown-4', 'value'),
    Input('mode-selector', 'value'),
    Input('start-month-dropdown', 'value'),
    Input('end-month-dropdown', 'value'),
    Input('checklist', 'value'),
    Input('my-range-slider', 'value')
)
def update_average_graph(data_json, dataset, selected_analyte, selected_analyte_2, mode, start_month, end_month, std_dev, year_range):
    df = pd.read_json(data_json, orient='split')

    def process_data(selected_analyte):
        temp_df = df.copy()
        if dataset == 'sub':
            #making correct flag key for specific analyte
            base_name = selected_analyte.split('_conc')[0]
            flag = f"{base_name}_flag"
            
            if selected_analyte in no_flag_list: #nss columns have no associated flag column 
                temp_df = temp_df.dropna(subset=[selected_analyte]) #dropping nan values for nss compounds in order to make regression line
            else:
                #setting conc to zero if flag column says below detection limit
                temp_df.loc[temp_df[flag] == 'BDL', selected_analyte] = 0
                #only keeping rows if there is a conc greater than zero or if i manually set conc to be zero (aka removing rows where conc is flagged for volume or other reasons)
                temp_df = temp_df[(temp_df[flag] == 'BDL') | (df[selected_analyte] > 0)]
        if dataset == 'super':
             #making correct flag key for specific analyte
            base_name = selected_analyte.split('_conc')[0]
            flag = f"{base_name}_flag"
            
            if selected_analyte in no_flag_list_super: #nss columns have no associated flag column 
                temp_df = temp_df.dropna(subset=[selected_analyte]) #dropping nan values for nss compounds in order to make regression line
            else:
                #setting conc to zero if flag column says below detection limit
                temp_df.loc[temp_df[flag] == 'BDL', selected_analyte] = 0
                #only keeping rows if there is a conc greater than zero or if i manually set conc to be zero (aka removing rows where conc is flagged for volume or other reasons)
                temp_df = temp_df[(temp_df[flag] == 'BDL') | (df[selected_analyte] > 0)]
        
        #filtering by year
        temp_df = temp_df[(temp_df['year'] >= year_range[0]) & (temp_df['year'] <= year_range[1])]
        return temp_df
            
    df_main = process_data(selected_analyte)

    if mode == 'seasonal':
        #filtering for selected months to average over
        df_main = df_main[(df_main['month'] >= start_month) & (df_main['month'] <= end_month)]
        if df_main.empty:
            return px.scatter(title="No data for selected months")
        grouped_main = df_main.groupby('year')[selected_analyte].agg(['mean', 'std']).reset_index()
        grouped_main.columns = ['date', 'avg_concentration', 'std_dev']
        x_main = grouped_main['date'].values

    else:  # monthly
        df_main['year_month'] = df_main['datetime'].dt.to_period('M').dt.to_timestamp()
        grouped_main = df_main.groupby('year_month')[selected_analyte].agg(['mean', 'std']).reset_index()
        grouped_main.columns = ['date', 'avg_concentration', 'std_dev']
        x_main = grouped_main['date'].astype(np.int64) // 10**9  # convert datetime to numeric (UNIX seconds)
        
    y_main = grouped_main['avg_concentration'].values
    
    # Regression line for primary analyte
    try:
        slope, intercept, *_ = theilslopes(y_main, x_main, 0.95)
        reg_line_main = intercept + slope * x_main
        slope_str_main = f"Slope: {slope:.4g}"
    except Exception:
        slope_str_main = "Slope: N/A"
        reg_line_main = None

    # Build figure
    fig = go.Figure()

    if std_dev:
        #adding first analyte and error bars
        fig.add_trace(go.Scatter(
            x=grouped_main['date'], 
            y=grouped_main['avg_concentration'], 
            mode='markers', 
            name=selected_analyte, 
            yaxis='y1', 
            marker=dict(color='blue'),
            error_y=dict(
                type='data',
                array=grouped_main['std_dev'],
                visible=True,
                thickness=1,
                width=3
            ),
        ))
    else:
        #adding first analyte selected to graph
        fig.add_trace(go.Scatter(
            x=grouped_main['date'], 
            y=grouped_main['avg_concentration'], 
            mode='markers', 
            name=selected_analyte, 
            yaxis='y1', 
            marker=dict(color='blue')
        ))

    #adding first regression line
    if reg_line_main is not None:
        fig.add_trace(go.Scatter(x=grouped_main['date'], y=reg_line_main, mode='lines', name=f"{selected_analyte} Theil-Sen Regression", line=dict(color='blue'), yaxis='y1'))

    # If a second analyte is selected
    if selected_analyte_2:
        df_2 = process_data(selected_analyte_2)

        if mode == 'seasonal':
            df_2 = df_2[(df_2['month'] >= start_month) & (df_2['month'] <= end_month)]
            grouped_2 = df_2.groupby('year')[selected_analyte_2].agg(['mean', 'std']).reset_index()
            grouped_2.columns = ['date', 'avg_concentration', 'std_dev']
            x_2 = grouped_2['date'].values
        else:
            df_2['year_month'] = df_2['datetime'].dt.to_period('M').dt.to_timestamp()
            grouped_2 = df_2.groupby('year_month')[selected_analyte_2].agg(['mean', 'std']).reset_index()
            grouped_2.columns = ['date', 'avg_concentration', 'std_dev']
            x_2 = grouped_2['date'].astype(np.int64) // 10**9

        y_2 = grouped_2['avg_concentration'].values

        #Regression line for secondary species
        try:
            slope2, intercept2, *_ = theilslopes(y_2, x_2, 0.95)
            reg_line_2 = intercept2 + slope2 * x_2
            slope_str_2 = f"Slope: {slope2:.4g}"
        except Exception:
            slope_str_2 = "Slope: N/A"
            reg_line_2 = None

        if std_dev:
            #adding second analyte to plot with error bars
            fig.add_trace(go.Scatter(
                x=grouped_2['date'], 
                y=grouped_2['avg_concentration'], 
                mode='markers', 
                name=selected_analyte_2, 
                yaxis='y2', 
                marker=dict(color='red'),
                error_y=dict(
                    type='data',
                    array=grouped_main['std_dev'],
                    visible=True,
                    thickness=1,
                    width=3
                ),
            ))
        else:
            #adding second analyte to plot
            fig.add_trace(go.Scatter(
                x=grouped_2['date'], 
                y=grouped_2['avg_concentration'], 
                mode='markers', 
                name=selected_analyte_2, 
                yaxis='y2', 
                marker=dict(color='red'))
            )

        #adding second regression line to plot
        if reg_line_2 is not None:
            fig.add_trace(go.Scatter(
                x=grouped_2['date'], y=reg_line_2, mode='lines', name=f"{selected_analyte_2} Theil-Sen Regression", line=dict(color='red'), yaxis='y2'))

        fig.update_layout(
            yaxis2=dict(
                title=selected_analyte_2,
                tickfont=dict(color='red'),
                overlaying='y',
                side='right'
            ),
        )
    subtitle_parts = [slope_str_main]
    if selected_analyte_2:
        subtitle_parts.append(slope_str_2)
    slope_subtitle = " | ".join(subtitle_parts)

    fig.update_layout(
        title=dict(
        text=f"{selected_analyte} (and {selected_analyte_2}) - {mode.title()} Average<br><sup>{slope_subtitle}</sup>",
        x=0.5
        ),
        xaxis_title="Year",
        yaxis=dict(
            title=selected_analyte,
            tickfont=dict(color='blue')
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=60, r=60, t=50, b=50)
    )
    return fig

@app.callback( 
    Output('ratio-graph', 'figure'),
    Input('stored-data', 'data'),
    Input('numerator-dropdown', 'value'),
    Input('denominator-dropdown', 'value')
)
def update_ratio_graph(data_json, numerator, denominator):
    df = pd.read_json(data_json, orient='split')
    
    df['ratio'] = df[numerator].div(df[denominator]) #making a new ratio column based on user choices
    df_filt = df.dropna(subset=['ratio']) #removing nan values
    df_filt = df_filt[df_filt['ratio'] != 0] #removing zero values

    #building figure 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filt['time (UTC)'], y=df_filt['ratio'], mode='markers', name=f'{numerator}/{denominator}'))

    fig.update_layout(
        title=f'{numerator}/{denominator} over Time',
        title_x=0.5,
        xaxis_title='Time (UTC)',
        yaxis_title='Ratio',
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=60, r=60, t=50, b=100),
        annotations=[
            dict(
                text="Note: Instances where the ratio was 0 or NaN were removed.",
                xref='paper', yref='paper',
                x=0, y=-0.2,  # Position below the plot
                showarrow=False,
                font=dict(size=12, color="gray"),
                align='left'
            )
        ],
    )
    return fig

@app.callback(
    Output('custom-months-container', 'style'),
    Input('mode-selector', 'value')
)
def toggle_custom_months(mode):
    if mode == 'seasonal':
        return {'display': 'block', 'marginBottom': '20px'}
    return {'display': 'none'}

if __name__ == '__main__':
    app.run(debug=True)
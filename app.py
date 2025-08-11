from dash import Dash, html, dcc, Input, Output
import pandas as pd
import datetime
from datetime import date
from erddapy import ERDDAP
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import theilslopes
import numpy as np

app = Dash()
g_server = app.server
#connecting to ERDDAP
server = "https://data.pmel.noaa.gov/pmel/erddap"
e = ERDDAP(server=server, protocol="tabledap", response="csv")
search_for = "station_barrow_submicron_chemistry"
end_url = e.get_search_url(search_for=search_for, response="csv")
dataset_ID = pd.read_csv(end_url)["Dataset ID"].iloc[0]
e.dataset_id = dataset_ID
df = e.to_pandas()

#data prep and calculations
df['datetime'] = pd.to_datetime(df['time (UTC)'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['nss_SO4_sub1_conc (micrograms m-3)'] = df['IC_total_SO4_sub1_conc (micrograms m-3)'] - (0.177*df['IC_Na_sub1_conc (micrograms m-3)'])
df['nss_K_sub1_conc (micrograms m-3)'] = df['IC_K_sub1_conc (micrograms m-3)'] - (0.02915*df['IC_Na_sub1_conc (micrograms m-3)'])
df['nss_Mg_sub1_conc (micrograms m-3)'] = df['IC_Mg_sub1_conc (micrograms m-3)'] - (0.088*df['IC_Na_sub1_conc (micrograms m-3)'])
df['nss_Ca_sub1_conc (micrograms m-3)'] = df['IC_Ca_sub1_conc (micrograms m-3)'] - (0.01441*df['IC_Na_sub1_conc (micrograms m-3)'])
df['Dust_conc (micrograms m-3)'] = (2.2*df['XRF_Al_sub1_conc (micrograms m-3)']) + (2.49*df['XRF_Si_sub1_conc (micrograms m-3)']) + (1.63*df['XRF_Ca_sub1_conc (micrograms m-3)']) + (2.42*df['XRF_Fe_sub1_conc (micrograms m-3)']) + (1.94*df['XRF_Ti_sub1_conc (micrograms m-3)'])
df['seasalt_sub1_conc (micrograms m-3)'] = df['IC_Cl_sub1_conc (micrograms m-3)'] + (df['IC_Na_sub1_conc (micrograms m-3)']*1.47)
#handling data flags and nan values
analytes = ['mass_sub1', 'IC_Na_sub1', 'IC_NH4_sub1', 'IC_K_sub1', 'IC_Mg_sub1', 'IC_Ca_sub1', 'IC_MSA_sub1', 'IC_Cl_sub1', 'IC_Br_sub1', 'IC_NO3_sub1', 'IC_total_SO4_sub1', 'IC_Oxalate_sub1', 'nss_SO4_sub1', 'nss_K_sub1',
'nss_Mg_sub1', 'nss_Ca_sub1', 'XRF_Na_sub1', 'XRF_Mg_sub1', 'XRF_Al_sub1', 'XRF_Si_sub1', 'XRF_P_sub1', 'XRF_S_sub1', 'XRF_Cl_sub1', 'XRF_K_sub1', 'XRF_Ca_sub1', 'XRF_Ti_sub1', 'XRF_V_sub1', 
'XRF_Cr_sub1', 'XRF_Mn_sub1', 'XRF_Fe_sub1', 'XRF_Co_sub1', 'XRF_Ni_sub1', 'XRF_Cu_sub1', 'XRF_Zn_sub1', 'XRF_Ga_sub1', 'XRF_Ge_sub1', 'XRF_As_sub1', 'XRF_Se_sub1', 'XRF_Br_sub1', 'XRF_Rb_sub1', 'XRF_Sr_sub1', 'XRF_Y_sub1', 'XRF_Zr_sub1',
'XRF_Mo_sub1', 'XRF_Pd_sub1', 'XRF_Ag_sub1', 'XRF_Cd_sub1', 'XRF_In_sub1', 'XRF_Sn_sub1', 'XRF_Sb_sub1', 'XRF_Ba_sub1', 'XRF_La_sub1', 'XRF_Hg_sub1', 'XRF_Pb_sub1', 'Dust', 'seasalt_sub1']

#list to use later bc nss values have no associated flag columns
no_flag_list = ['nss_SO4_sub1_conc (micrograms m-3)', 'nss_K_sub1_conc (micrograms m-3)', 'nss_Mg_sub1_conc (micrograms m-3)', 'nss_Ca_sub1_conc (micrograms m-3)', 'Dust_conc (micrograms m-3)', 'seasalt_sub1_conc (micrograms m-3)']

#columns that i dont want to show up in the drop down options
#exclude_col = ['time (UTC)', 'latitude (degrees_north)', 'longitude (degrees_east)', 'Stop_time (UTC)', 'Station', 'Filter_ID', 'volume (m3)', 'datetime', 'year', 'month']
app.layout = html.Div([
    html.Div(
        html.H1('Barrow Observatory Data Display'), # Page title
        style={'textAlign': 'center', 'marginBottom': '30px'}
    ),
    html.Div(
        html.H2('Plot All Data'),
        style={'textAlign': 'left', 'marginBottom': '15px'}
    ),
    html.Label('Select a variable to display:'),
    dcc.Dropdown( #first dropdown for graph 1
        id='analyte-dropdown',
        options = [
            {'label': f"{analyte}_conc (micrograms m-3)", 'value': f"{analyte}_conc (micrograms m-3)"}
            for analyte in analytes
        ],
        value='mass_sub1_conc (micrograms m-3)',
        clearable=False
    ),
    html.Label('Select a second variable to display:'),
    dcc.Dropdown( #first dropdown for graph 1
        id='analyte-dropdown-2',
        options=[
            {'label': f"{analyte}_conc (micrograms m-3)", 'value': f"{analyte}_conc (micrograms m-3)"}
            for analyte in analytes
        ],
        value=None,  # default for second analyte
        clearable=True
    ),
    dcc.Graph(
        id='all-data-graph',
        style={'width': '80%', 'height': '600px', 'marginLeft':'auto', 'marginRight':'auto'}
    ), # graph 1
    html.Div(
        html.H2('Plot Data Averages'),
        style={'textAlign': 'left', 'marginBottom': '15px'}
    ),
    html.Label('Select a variable to display:'),
    dcc.Dropdown( #first dropdown for graph 2
        id='analyte-dropdown-3',
        options = [
            {'label': f"{analyte}_conc (micrograms m-3)", 'value': f"{analyte}_conc (micrograms m-3)"}
            for analyte in analytes
        ],
        value='mass_sub1_conc (micrograms m-3)',
        clearable=False
    ),
    html.Label('Select a second variable to display:'),
    dcc.Dropdown( #second dropdown for graph 2
        id='analyte-dropdown-4',
        options=[
            {'label': f"{analyte}_conc (micrograms m-3)", 'value': f"{analyte}_conc (micrograms m-3)"}
            for analyte in analytes
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
        html.H2('Species Ratios over Time Plot'),
        style={'textAlign': 'left', 'marginBottom': '15px'}
    ),
    html.Label('Select a variable to go in the numerator:'),
    dcc.Dropdown( #first dropdown for graph 3
        id='numerator-dropdown',
        options = [
            {'label': f"{analyte}_conc (micrograms m-3)", 'value': f"{analyte}_conc (micrograms m-3)"}
            for analyte in analytes
        ],
        value='mass_sub1_conc (micrograms m-3)',
        clearable=False
    ),
    html.Label('Select a variable to go in the denominator:'),
    dcc.Dropdown( #second dropdown for graph 3
        id='denominator-dropdown',
        options = [
            {'label': f"{analyte}_conc (micrograms m-3)", 'value': f"{analyte}_conc (micrograms m-3)"}
            for analyte in analytes
        ],
        value='IC_Na_sub1_conc (micrograms m-3)',
        clearable=False
    ),
     dcc.Graph( # graph 3
        id='ratio-graph', 
        style={'width': '80%', 'height': '600px', 'marginLeft':'auto', 'marginRight':'auto'}
        ), 
])
@app.callback( 
    Output('all-data-graph', 'figure'),
    Input('analyte-dropdown', 'value'),
    Input('analyte-dropdown-2', 'value')
)
def update_graph(selected_analyte, selected_analyte_2):
    fig = go.Figure()

    # adding first variable to graph
    fig.add_trace(go.Scatter(x=df['time (UTC)'], y=df[selected_analyte], mode='markers', name=selected_analyte, yaxis='y1'))

# Add second trace only if selected
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
        title="Comparison of Two Analytes Over Time" if selected_analyte_2 else f"{selected_analyte} Over Time",
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

@app.callback(
    Output('average-graph', 'figure'),
    Input('analyte-dropdown-3', 'value'),
    Input('analyte-dropdown-4', 'value'),
    Input('mode-selector', 'value'),
    Input('start-month-dropdown', 'value'),
    Input('end-month-dropdown', 'value'),
    Input('checklist', 'value'),
    Input('my-range-slider', 'value')
)
def update_average_graph(selected_analyte, selected_analyte_2, mode, start_month, end_month, std_dev, year_range):
    def process_data(selected_analyte):
        temp_df = df.copy()

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
    Input('numerator-dropdown', 'value'),
    Input('denominator-dropdown', 'value')
)
def update_ratio_graph(numerator, denominator):
    df_copy = df.copy()
    
    df_copy['ratio'] = df[numerator].div(df[denominator]) #making a new ratio column based on user choices
    df_filt = df_copy.dropna(subset=['ratio']) #removing nan values
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
'''
    # Theil-Sen regression
    try:
        slope, intercept, *_ = theilslopes(y, x, 0.95)
        reg_line = intercept + slope * x
        slope_str = f"Slope: {slope:.4g}"
    except Exception:
        slope_str = "Slope: N/A"
        reg_line = None

    # Build figure
    fig = px.scatter(grouped, x='date', y='avg_concentration',
                     title=f"{selected_analyte} - {mode.title()} Average<br><sup>{slope_str}</sup>")

    fig.update_layout(title_x=0.5)

    if reg_line is not None:
        fig.add_scatter(x=grouped['date'], y=reg_line, mode='lines',
                        name='Theil-Sen Regression', line=dict(color='red'))
    return fig
'''

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
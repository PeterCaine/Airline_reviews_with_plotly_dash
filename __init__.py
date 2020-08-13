from flask import Flask, send_from_directory 
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import os
import pickle
import random

from utils import load_dfs, keyword_reviews, write_out, freq_bigram_finder, freq_trigram_finder, \
    count_patterns, open_unique_adjn, load_adj_count, df_lookup_iloc, load_unique, \
    review_ids_for_keyword, airline_bucket_reviews_count, graph_display, reviews_on_single_keyword, \
    context_filter, on_brand_text
# ------------------------------------------------------------------------


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = Flask(__name__) 
app = dash.Dash(server=server, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    # URL
    dcc.Location(id='url', refresh=True),
    # HEADER
    html.Div([
        html.Img(
            id='airline_logo1',
            src=app.get_asset_url('ghost_TA.png'),
            style={
                'width': '11%',
                # 'float': 'right',
                # 'position': 'relative',
                'margin-top': 40,
                'margin-left': 20
            },
            className='two columns'
        ),
        html.H1(
            children="Brand Analysis",
            style={
                'text-align': 'right',
                'margin-top': 50
            },
            className='five columns'
        ),
        html.H5(
            children="Airline reviews",
            style={
                'text-align': 'left',
                'margin-top': 70,
                'margin-left': 10,
                'text-decoration': 'underline overline'
            },
            className='three columns'
        ),
        html.Img(
            id='airline_logo2',
            src=app.get_asset_url('ghost_TA.png'),
            # className = 'three columns',
            style={
                # 'height':"8.33%",
                'width': "11%",
                'float': 'right',
                # 'position':'relative',
                'margin-top': 40,
                'margin-right': 20
            },
            className='two columns'
        )
    ], id='header', className='row'),
    html.Hr(),
    dcc.Tabs(id='mother_tab', value='tab_1', children=[
        dcc.Tab(value='tab_1', label='Main', children=[
                # BODY
                html.Div([
                    # NAVBAR
                    html.Div([
                        html.Div([
                            html.Br(),
                            html.H5(children="Select the airline"),
                            dcc.Dropdown(
                                id='airline_dropdown',
                                options=[
                                    {'label': 'British Airways', 'value': 'British_Airways'},
                                    {'label': 'EasyJet', 'value': 'EasyJet'},
                                    {'label': 'KLM', 'value': 'KLM'},
                                    {'label': 'Ryanair', 'value': 'Ryanair'},
                                    {'label': 'Virgin Atlantic', 'value': 'Virgin'}
                                ],
                                value='KLM'
                            )
                        ]),
                        html.Br(),
                        html.H5("Select ratings range"),
                        dcc.RangeSlider(
                            id='my_range_slider',
                            min=1,
                            max=5,
                            step=None,
                            value=[1, 5],
                            marks={
                                1: '1 star',
                                2: '2 star',
                                3: '3 star',
                                4: '4 star',
                                5: '5 star'
                            },
                        ),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.Table([
                            html.Thead([html.H5("Review Stats")]),
                            html.Tbody([
                                html.Tr([
                                    html.Td("total number of texts"),
                                    html.Td(id='total_texts')
                                ]),
                                html.Tr([
                                    html.Td("total number in rating range"),
                                    html.Td(id='total_in_range')
                                ]),
                                html.Tr([
                                    html.Td("total returned"),
                                    html.Td(id='total_returned')
                                ]),
                                html.Tr([
                                    html.Td("percent returned"),
                                    html.Td(id='percent_texts', style={
                                        'background': 'green', 'color': 'white'})
                                ])
                            ])
                        ], style={'width': '100%'}, className='table'),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.H5(children="General Linguistic Features"),
                        html.Button(id='bigrams_btn', children="Signature bigrams (PMI)",
                                    style={'width': '100%'}),
                        html.Br(),
                        html.Button(id='trigrams_btn', children="Signature trigrams (PMI)",
                                    style={'width': '100%'}),
                        html.Br(),
                        html.Button(id='common_btn', children="Most Common",
                                    style={'width': '100%'}),
                        html.Br(),
                        html.Button(id='unique_btn', children="Unique Phrases",
                                    style={'width': '100%'}),
                        html.Br(),
                        html.Button(id='visualise_btn', children="Visualise Clusters",
                                    style={'width': '100%'}),
                        html.Br(),
                        html.Button(id='variation_btn', children="Varied Reviews",
                                    style={'width': '100%'}),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                    ], id='navbar', className='three columns'),
                    html.Br(),
                    # SELECTOR ROW - this was proving too slow on the server so have omitted this in production
                    html.Div([
                        # html.Div([
                        #     html.Br(),
                        #     html.Br(),
                        #     dcc.RadioItems(
                        #         id='synonym_selector',
                        #         options=[
                        #             {'label': 'Use Synonyms', 'value': 'y'},
                        #             {'label': 'No Synonyms', 'value': 'n'}
                        #         ],
                        #         value='n'),
                        # ], className='two columns'),
                        html.Div([
                            html.Div([
                                html.H5(children='Keywords used:'),
                                html.P(
                                    id='keyword_list'
                                ),
                            ],
                                className='six columns'
                            ),
                        ]),
                        html.Div([
                            html.H5(children="Enter Keywords"),
                            dcc.Textarea(
                                id='keyword_input',
                                placeholder="Enter keywords separated by commas or review number prefixed with '#'",
                                value='',
                                cols=70
                            )
                        ], style={'float': 'right'})
                    ], className='nine columns', style={'left-margin': 50}),
                    html.Br(),
                    html.Hr(className='seven columns offset-by-one'),
                    # GRAPH
                    html.Div([
                        dcc.Loading(id='main-loadscreen', children=[
                            dcc.Graph(
                                id='keyword_contribution',
                                figure={}
                            ),
                        ]),
                        html.Br(),
                        html.Br(),
                        html.Div([
                            html.H5("Analysis Out"),
                            dcc.Loading([
                                html.P(
                                    id='analysis_text',
                                )
                            ]),
                        ], className='four columns'),
                        # TEXT FIELD
                        html.Div([
                            html.H5(children='Sample review:'),
                            html.P(id='main_text_container'),
                        ], className='seven columns'),
                        html.Br(),
                        html.Br(),
                        html.Button(id='save_texts',
                                    children="Save Texts",
                                    style={'width': '10%', 'float': 'right', 'position': 'relative'},
                                    title="After saving, download the saved texts in the link above."
                                    ),
                        html.Br(),
                        html.Br(),
                    ], id='right_pane', className='nine columns'),
                    html.Div(id='empty_div'),
                    html.A(id='download_saved_texts', children="Download Saved Texts",
                           href='/downloads/', target='_blank', title="Only the last saved texts will be downloaded")
                ], id='body', className='row'),
                ], className='custom-tab', selected_className='custom-tab--selected'),  # we modified here
        dcc.Tab(id='bucket_tab', value='tab_bucket', label='On-brand Evaluator', children=[
                html.Br(),
                html.Br(),
                html.Div([
                    html.H5("Each airline proposes characteristics of the brand."),
                    html.Div("""For each airline, these characteristics have been condensed into concepts and each
                    concept associated with lists of keywords.
                    Reviews which mention these keywords are tallied and compared.

                    These keywords can be explored in the 'explore keywords' tab.""", style={'margin-left': 100}),
                ], className='eight columns offset-by-two'),
                html.Div(id='none'),
                html.Br(),
                html.Br(),
                html.Div([
                    html.Br(),
                    html.Br(),
                    html.H5("Select the airline"),
                    dcc.Dropdown(
                        id='airline_in_focus',
                        options=[
                            {'label': 'British Airways', 'value': 'British_Airways'},
                            {'label': 'EasyJet', 'value': 'EasyJet'},
                            {'label': 'KLM', 'value': 'KLM'},
                            {'label': 'Ryanair', 'value': 'Ryanair'},
                            {'label': 'Virgin Atlantic', 'value': 'Virgin'}
                        ],
                        value='KLM',
                        className='two columns',
                    ),
                    html.Br(),
                    html.Br(),
                    dcc.Graph(id='in_focus_graph', className='ten columns offset-by-one'),
                    html.Br(),
                    html.Div(id='div_out', className='ten columns offset-by-one'),
                    html.Br(),
                    html.Hr(className='ten columns offset-by-one'),
                    dcc.Graph(id='BA_graph', className='five columns'),
                    dcc.Graph(id='EZ_graph', className='five columns'),
                    dcc.Graph(id='KL_graph', className='five columns'),
                    dcc.Graph(id='RY_graph', className='five columns'),
                    dcc.Graph(id='VI_graph', className='five columns')
                ])
                ], className='custom-tab', selected_className='custom-tab--selected'),  # AND HERE
        dcc.Tab(label='Explore Keywords', value='tab_concept', children=[
                html.Div(id='probe_display', children=[
                    html.Br(),
                    html.Br(),
                    html.H5(children="Input keyword for analysis"),
                    dcc.Input(id='keyword-to-compare',
                              value='exceptional',
                              placeholder="input keyword to compare",
                              className='two columns',
                              ),
                    dcc.Loading(id='loading-icon', children=[
                        dcc.Graph(id='compare_word',
                                  figure={},
                                  className='eight columns',
                                  ),
                    ]),
                ], className='row'),
                html.Br(),
                html.Br(),
                html.Div([
                    html.Div([
                        html.H5("Select airline"),
                        dcc.Dropdown(id='airlines_2',
                                     options=[{'label': 'British Airways', 'value': 'British_Airways'},
                                              {'label': 'EasyJet', 'value': 'EasyJet'},
                                              {'label': 'KLM', 'value': 'KLM'},
                                              {'label': 'Ryanair', 'value': 'Ryanair'},
                                              {'label': 'Virgin', 'value': 'Virgin'}
                                              ],
                                     value='KLM'
                                     ),
                    ], className='three columns'),
                    html.Div([
                        html.H5("Select text span"),
                        dcc.RadioItems(
                            id='concordance_select',
                            options=[
                                {'label': 'Immediate context', 'value': 'short'},
                                {'label': 'Full text', 'value': 'long'}
                            ],
                            value='short',
                            labelStyle={'display': 'inline-block'}
                        ),
                    ], className='three columns'),
                    html.Div([
                        html.H5("Filter context"),
                        dcc.Input(id='context_keyword_input',
                                  value='',
                                  placeholder="""Enter keyword to filter for specific bigrams. Separate multiple
keywords with a comma""",
                                  style={'width': '100%'},
                                  className='two columns'
                                  )
                    ], className="six columns"),
                ], className='row'),
                html.Br(),
                html.Br(),
                html.Div([
                    html.Br(),
                    html.Div(id='dt_1'),
                    html.Br(),
                    html.Br(),
                ], className='eight columns offset-by-two')
                ], className='custom-tab', selected_className='custom-tab--selected'), 
        dcc.Tab(label='Adjective Distribution', children=[
            html.Br(),
            html.Br(),
            html.Div([
                html.Div([
                    dcc.Slider(
                        id='num_adj',
                        min=20,
                        max=200,
                        step=10,
                        value=50,
                        marks={
                            20: {'label': 'fewer'},
                            200: {'label': 'more'}
                        },
                        className='nine columns'),
                    dcc.RadioItems(
                        id='unique_switch',
                        options=[{'label': 'Unique Only', 'value': 1},
                                 {'label': 'All', 'value': 0}],
                        value=0,
                        style={'float': 'right'}
                    ),
                ], className='four columns offset-by-four'),
            ], className='row'),
            html.Div([
                dcc.Graph(id='graph_BA'),
                dcc.Graph(id='graph_Easy'),
                dcc.Graph(id='graph_Klm'),
                dcc.Graph(id='graph_Ryan'),
                dcc.Graph(id='graph_Virgin')
            ], className='twelve columns ')
        ], value='tab_2',
            className='custom-tab', selected_className='custom-tab--selected'), 
    ]),
], className='ten columns offset-by-one')
# ------------------------------------------------------------------------------------------------


@app.callback(
    [Output(component_id='main_text_container', component_property='children'),
     Output(component_id='keyword_list', component_property='children'),
     Output(component_id='keyword_contribution', component_property='figure'),
     Output(component_id='percent_texts', component_property='children'),
     Output(component_id='total_returned', component_property='children'),
     Output(component_id='total_in_range', component_property='children'),
     Output(component_id='total_texts', component_property='children'),
     Output("download_saved_texts", "href")
     ],
    [Input('airline_dropdown', 'value'),
     # Input('synonym_selector', 'value'),
     Input('my_range_slider', 'value'),
     Input('save_texts', 'n_clicks'),
     Input('keyword_input', 'value'),
     Input('variation_btn', 'n_clicks'),
     Input("download_saved_texts", "n-clicks")]
)
def update_output(airline_dropdown,  my_range_slider,
                  save_texts, keyword_input, variation_btn, download_saved_texts):  # synonym_selector
    image = app.get_asset_url(f'{airline_dropdown}.png')
    keyword_dict = pickle.load(open('./data/keyword_dict.pkl', 'rb'))
    override_txt = 0
    if keyword_input == "":
        keywords = keyword_dict[airline_dropdown]
    elif keyword_input.startswith('#'):
        review_idx = keyword_input.lstrip('#')
        override_txt = df_lookup_iloc(airline_dropdown, int(review_idx))
        keywords = []
    else:
        keywords = keyword_input.split(',')
        keywords = [word.strip() for word in keywords]
        if keywords[-1] == '':
            keywords.pop()
    df, tot, abs_tot = load_dfs(
        airline_dropdown, min_rating=my_range_slider[0], max_rating=my_range_slider[1])
    texts, plotly, updated_keywords, overlaps = keyword_reviews(
        keywords, df)  # deleted use_syn=synonym_selector
    keyword_text = ', '.join(updated_keywords)
    fig = go.Figure(data=[go.Bar(
                            x=list(plotly.keys()),
                            y=list(plotly.values())
                    )],
                    layout=go.Layout(
                        title=go.layout.Title(text="How Keywords contribute to Reviews"),
                        template='plotly_white'
                    )
    )
    href_out = os.path.join(
        'downloads',
        f'{airline_dropdown}_texts.txt')
    if dash.callback_context.triggered:
        changed_id = dash.callback_context.triggered[0]
        if 'save_texts' in changed_id['prop_id']:
            write_out(airline_dropdown, texts)
            relative_filename = os.path.join( 
                'downloads',  f'{airline_dropdown}_texts.txt')
            href_out = f'/{relative_filename}'  
        elif 'variation' in changed_id['prop_id']:
            fig = go.Figure(data=[go.Bar(x=[tup[0] for tup in overlaps],
                             y=[tup[2] for tup in overlaps])],
                            layout=go.Layout(
                                title='High variation texts'
                            )
            )

    try:
        text = random.choice(texts)
        text_out = 'SAMPLE TEXT: \n' + text
    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        text_out = "There are NO REVIEWS containing this keyword"

    num_returned = len(texts)
    percent = str(round(num_returned/tot*100, 2)) + ' %'
    percent_out = f'{percent}'
    if override_txt == 0:
        container = text_out
    else:
        container = 'Review: ' + str(review_idx) + ": " + override_txt

    fig.update_layout(yaxis_title='Number of Reviews')
    return container, keyword_text, fig, percent_out, num_returned, tot, abs_tot, href_out


@app.callback(
    Output(component_id='analysis_text', component_property='children'),
    [Input('airline_dropdown', 'value'),
     Input('bigrams_btn', 'n_clicks'),
     Input('trigrams_btn', 'n_clicks'),
     Input('common_btn', 'n_clicks'),
     Input('unique_btn', 'n_clicks'),
     Input('my_range_slider', 'value')
     ]
)
def analysis_section(airline_dropdown, bigrams_btn, trigrams_btn, common_btn, unique_btn, my_range_slider):
    df, *_ = load_dfs(airline_dropdown, min_rating=my_range_slider[0], max_rating=my_range_slider[1])
    num_return = 50
    if dash.callback_context.triggered:
        changed_id = dash.callback_context.triggered[0]
        if 'bigrams' in changed_id['prop_id']:
            analysis_list = freq_bigram_finder(df, num_return=num_return)
            analysis_text = ' - '.join(analysis_list)
            return "BIGRAMS: "+analysis_text
        elif 'trigrams' in changed_id['prop_id']:
            analysis_list = freq_trigram_finder(df, num_return=num_return)
            analysis_text = ' - '.join(analysis_list)
            return "TRIGRAMS: "+analysis_text
        elif 'common' in changed_id['prop_id']:
            analysis_list = count_patterns(df, num_return=num_return)
            analysis_text = ' - '.join(analysis_list)
            return "MOST COMMON: "+analysis_text
        elif 'unique' in changed_id['prop_id']:
            analysis_text = open_unique_adjn(airline_dropdown, num_return=num_return)
            return "UNIQUE PHRASES: "+analysis_text


@app.callback(
    Output('url', 'href'),
    [Input('visualise_btn', 'n_clicks'),
     Input('airline_dropdown', 'value')]
)
def navigate(visualise_btn, airline_dropdown):
    if dash.callback_context.triggered:
        changed_id = dash.callback_context.triggered[0]
        if 'visualise' in changed_id['prop_id']:
            filepath = f'static/ldavis_{airline_dropdown}15.html'
            return filepath


@app.callback(
    [Output('graph_BA', 'figure'),
     Output('graph_Easy', 'figure'),
     Output('graph_Klm', 'figure'),
     Output('graph_Ryan', 'figure'),
     Output('graph_Virgin', 'figure')
     ],
    [Input('num_adj', 'value'),
     Input('unique_switch', 'value')]
)
def display_adj_count(slider, unique_switch):
    val = slider
    fig_ba = load_adj_count('British_Airways', val)
    fig_easy = load_adj_count('EasyJet', val)
    fig_klm = load_adj_count('KLM', val)
    fig_ryanair = load_adj_count('Ryanair', val)
    fig_virgin = load_adj_count('Virgin', val)
    if unique_switch == 0:
        return fig_ba, fig_easy, fig_klm, fig_ryanair, fig_virgin
    else:
        out_list = []
        out_dict = load_unique(val)
        for key, value in out_dict.items():
            x = [tup[0] for tup in value]
            y = [tup[1] for tup in value]
            fig = px.bar(x=x, y=y, color=y, title=f"Most Common Unique Adjectives: {key}",
                         template='plotly_white')
            out_list.append(fig)
        return out_list[0], out_list[1], out_list[2], out_list[3], out_list[4]


@app.callback(
    Output('compare_word', 'figure'),
    [Input('keyword-to-compare', 'value')]
)
def compare_keywords(keyword):
    airlines = ['British_Airways', 'EasyJet', 'KLM', 'Ryanair', 'Virgin']
    keyword = keyword.strip()
    kwd = [keyword]
    fig = go.Figure(data=[
        go.Bar(name=airlines[0], x=kwd, y=[review_ids_for_keyword(airlines[0], keyword) / 148]),
        go.Bar(name=airlines[1], x=kwd, y=[review_ids_for_keyword(airlines[1], keyword) / 148]),
        go.Bar(name=airlines[2], x=kwd, y=[review_ids_for_keyword(airlines[2], keyword) / 148]),
        go.Bar(name=airlines[3], x=kwd, y=[review_ids_for_keyword(airlines[3], keyword) / 148]),
        go.Bar(name=airlines[4], x=kwd, y=[review_ids_for_keyword(airlines[4], keyword) / 145])
    ])
    fig.update_layout(template='plotly_white', colorway=['blue', 'orange', 'lightblue', 'darkblue', 'red'],
                      yaxis_title="% texts containing word")
    return fig


@app.callback(
    Output('dt_1', 'children'),
    [Input('keyword-to-compare', 'value'),
     Input('airlines_2', 'value'),
     Input('concordance_select', 'value'),
     Input('context_keyword_input', 'value')
     ]
)
def reveal_datatable(keyword, airlines_2, concordance_select, context_keyword_input):
    df, *_ = load_dfs(airlines_2)
    df_out = reviews_on_single_keyword(keyword, df)
    if concordance_select == 'long':
        data, columns, _ = context_filter(df_out, keyword, keyword_list=context_keyword_input)
        return dt.DataTable(style_cell={'whiteSpace': 'normal', 'height': 'auto',
                                        'textAlign': 'left', 'fontSize': 18,
                                        'font-family': 'sans-serif'}, data=data, columns=columns)
    else:
        *_, out_text = context_filter(df_out, keyword, keyword_list=context_keyword_input)
        str_out = ''
        for line in out_text:
            str_out += ' '.join(line[0])[-50:].lstrip() + '        ' + '**' + line[1] + '**' + '        ' + ' '.join(
                line[2]) + '\n'
        return dcc.Markdown(str_out, style={"white-space": "pre", 'font-size': 18})


@app.callback(
    [Output('BA_graph', 'figure'),
     Output('EZ_graph', 'figure'),
     Output('KL_graph', 'figure'),
     Output('RY_graph', 'figure'),
     Output('VI_graph', 'figure')
     ],
    [Input('none', 'children')]
)
def pop_bucket_graph(airline_in_focus):
    out_dict = airline_bucket_reviews_count('British_Airways')
    fig1 = graph_display(out_dict, 'British_Airways')
    out_dict = airline_bucket_reviews_count('EasyJet')
    fig2 = graph_display(out_dict, 'EasyJet')
    out_dict = airline_bucket_reviews_count('KLM')
    fig3 = graph_display(out_dict, 'KLM')
    out_dict = airline_bucket_reviews_count('Ryanair')
    fig4 = graph_display(out_dict, 'Ryanair')
    out_dict = airline_bucket_reviews_count('Virgin')
    fig5 = graph_display(out_dict, 'Virgin')

    return fig1, fig2, fig3, fig4, fig5


@app.callback(
    [Output('in_focus_graph', 'figure'),
     Output('div_out', 'children')],
    [Input('airline_in_focus', 'value'),
     Input('BA_graph', 'figure'),
     Input('EZ_graph', 'figure'),
     Input('KL_graph', 'figure'),
     Input('RY_graph', 'figure'),
     Input('VI_graph', 'figure')]
)
def for_div_out(airline_in_focus, BA_graph, EZ_graph, KL_graph, RY_graph, VI_graph):
    tups_out = on_brand_text(airline_in_focus)
    str_out = '### Keywords per category used: \n'
    for line in tups_out:
        str_out += '* **' + line[0] + ': ' + '**' + line[1] + '\n'

    if airline_in_focus == 'British_Airways':
        return BA_graph, dcc.Markdown(str_out, style={"white-space": "pre"})
    elif airline_in_focus == 'EasyJet':
        return EZ_graph, dcc.Markdown(str_out, style={"white-space": "pre"})
    elif airline_in_focus == 'KLM':
        return KL_graph, dcc.Markdown(str_out, style={"white-space": "pre"})
    elif airline_in_focus == 'Ryanair':
        return RY_graph, dcc.Markdown(str_out, style={"white-space": "pre"})
    else:
        return VI_graph, dcc.Markdown(str_out, style={"white-space": "pre"})


@server.route("/downloads/<path:path>")
def download(path):
    """Serve a file from the downloads directory."""
    root_dir = os.getcwd()
    return send_from_directory(
        os.path.join(root_dir, 'downloads'), path
    )


if __name__ == "__main__":
    app.run_server()

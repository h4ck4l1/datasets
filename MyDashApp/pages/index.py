from dash import html,dcc,register_page
import dash_bootstrap_components as dbc

first_msg = dcc.Markdown(
    '''

    ## Observations and Experiments carried:

    ➼ Fitting tests were done on all the independent variables of each dataset.

    ➼ Top 10 Best Fits were plotted with interactive graphs.

    ➼ Inferences were drawn on each variable and the strength of the fit and fit was weighed for further explanations.

    ➼ Machine Learning was implemented for relevant datasets.

    ➼ Classification for Mushroom, Time Series analysis for kfc-stock, and classification with geo location for covid-19.

    ➼ Different ML algorithms were used ranging from simple Linear Regression to complex Gradient Boosting bagging algorithms and Neural Nets.

    ➼ For every dataset an index page has been provided for feature descrptions, experimentation methods, data collection strategies, cleaning processes etc.,

    ''',
    className="subpage-markdown",
)

second_msg = dcc.Markdown(
    '''
    ## Steps Followed

    ### Data Collection
    #### ⬇
    ### Data Preprocessing
    #### ⬇
    ### Model Training
    #### ⬇
    ### Model Evaluation
    #### ⬇
    ### Results Interpretation
    ''',
    className="subpage-markdown-2"
)


layout = html.Div(
    [
        html.Div(html.H1("بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ",className="main-heading"),className="heading-divs"),
        first_msg,
        html.Div(second_msg,className="heading-divs"),
        html.H3("**Disclaimer: Web-Application might work different in different web browsers please switch to desktop version on android app if there are discrepancies and for best viewing experience use large screen",className="index-second-heading"),
    ],
    className="subpage-content"
)

register_page("INDEX",path="/",layout=layout)
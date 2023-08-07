import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates['mod'] = go.layout.Template(layout=dict(font=dict(family="Fira Code")))
pio.templates.default = "plotly_dark+mod"
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats
import scipy.stats as ss
from dash import Dash,html,dcc,Input,Output,callback,register_page,dash_table

salaries = pd.read_csv('https://github.com/h4ck4l1/datasets/raw/main/Normal_practice/ds_salaries.csv')


salaries['work_year'] = salaries.work_year.astype(str)

salaries.drop(['job_title','salary_currency','salary'],inplace=True,axis=1)

idx = pd.IndexSlice
years = salaries.work_year.unique()
exp_levels = salaries.experience_level.unique()
emp_types = salaries.employment_type.unique()

"""
## Message
"""

msgs = '''
The statistical analysis has been carried out here and explained in brief terms  
The tests used are  
- [Levene's](https://en.wikipedia.org/wiki/Levene%27s_test) test to test for equalitly of variances  
- [Oneway Anova](https://en.wikipedia.org/wiki/One-way_analysis_of_variance) to test for equality of means in k-groups but one variable  

'''

"""
## Mean Analysis
"""

Mean_Analysis = make_subplots(rows=2,cols=2,horizontal_spacing=0.2,vertical_spacing=0.2)
for i,year in enumerate(salaries.work_year.unique()):
    sal = salaries.query(f"work_year == '{year}'").salary_in_usd.to_numpy()
    sal = StandardScaler().fit_transform(sal.reshape(-1,1)).ravel()
    qq = sm.qqplot(sal,line='s').gca().lines
    plt.close()
    Mean_Analysis.add_trace(go.Scatter(x=qq[0].get_xdata(),y=qq[0].get_ydata(),mode="markers"),row=((i+1)//3)+1,col=(i%2)+1)
    Mean_Analysis.add_trace(go.Scatter(x=qq[1].get_xdata(),y=qq[1].get_ydata(),mode="lines"),row=((i+1)//3)+1,col=(i%2)+1)
    Mean_Analysis.add_annotation(text=f"{year}",font=dict(size=20),bordercolor="white",showarrow=False,x=0,xanchor="left",xref='x domain',y=1.1,yref='y domain',yanchor="bottom",row=((i+1)//3)+1,col=(i%2)+1)
    Mean_Analysis.add_annotation(text=f"count: {len(sal)}",font=dict(size=20),bordercolor="white",showarrow=False,x=0.8,xanchor="center",y=1.1,yref='y domain',yanchor="bottom",row=((i+1)//3)+1,col=(i%2)+1)

Mean_Analysis.update_layout(width=1000,height=700,showlegend=False)

year_2020 = salaries.query("work_year == '2020'").salary_in_usd.to_numpy()
year_2021 = salaries.query("work_year == '2021'").salary_in_usd.to_numpy()
year_2022 = salaries.query("work_year == '2022'").salary_in_usd.to_numpy()
year_2023 = salaries.query("work_year == '2023'").salary_in_usd.to_numpy()

mean_inf = '''
The Higher count Years are very much approximated to nomral plot but the 76 tends to skew.
As the lesser count plot is concave up w.r.t to the normal q-q line means that it is right skewed.
but as it can be assumed to that 2020 is also derived from normal population so we can apply manova test to see whether there is any statistical significance between the means
Doing variance test

- **levene's test**
- - we get pvalue=0.20411, we conclude that variances are equal
- **anova test**
- - we get pvalue = 9.9e-48, we conclude that there is significant difference in means


'''

from scipy.stats import levene

levene(year_2020,year_2021,year_2022,year_2023)

sm.stats.anova_oneway((year_2020,year_2021,year_2022,year_2023),use_var='equal')

"""
## Experience Level Mean Analysis
"""

sal = salaries.copy()
sal.set_index(['work_year','experience_level'],inplace=True)
sal.sort_index(inplace=True)
exp_level_mean = make_subplots(rows=4,cols=4)
sal.loc[sal.index.unique()[0]]
for i,year_exp in enumerate(sal.index.unique()):
    vals = StandardScaler().fit_transform(sal.loc[year_exp].salary_in_usd.to_numpy().reshape(-1,1)).ravel()
    data = sm.qqplot(vals,line='s').gca().lines
    plt.close()
    exp_level_mean.add_trace(go.Scatter(x=data[0].get_xdata(),y=data[0].get_ydata(),mode="markers"),row=(i//4)+1,col=(i%4)+1)
    exp_level_mean.add_trace(go.Scatter(x=data[1].get_xdata(),y=data[1].get_ydata(),mode="lines"),row=(i//4)+1,col=(i%4)+1)
    exp_level_mean.add_annotation(text=f"{year_exp}",font=dict(size=15),x=0.8,y=1.05,xanchor="center",yanchor="bottom",xref='x domain',yref='y domain',bordercolor="white",showarrow=False,row=(i//4)+1,col=(i%4)+1)
    exp_level_mean.add_annotation(text=f"Count :{len(data[0].get_xdata())}",font=dict(size=15),x=0.3,y=1.05,xanchor="center",yanchor="bottom",xref='x domain',yref='y domain',bordercolor="white",row=(i//4)+1,col=(i%4)+1,showarrow=False)

exp_level_mean.update_layout(height=800,hovermode=False,showlegend=False)

tab_vals = salaries.groupby(['work_year','experience_level']).aggregate({'salary_in_usd':'mean'}).unstack(level=1)
tab_vals.reset_index(inplace=True)

l = []
for item in tab_vals.to_dict('records'):
    d = {}
    for key,value in item.items():
        d[key[1]] = value
    l.append(d)

table1 = dash_table.DataTable(
        data=l,
        columns=[{'name': [col[0],col[1]], 'id': col[1]} for col in tab_vals.columns],
        merge_duplicate_headers=True,
        style_header={'textAlign':'center'},
        style_cell={'textAlign':'left'}
    )


exp_level_mean_inf = '''
Assumptions:  
  - Independent samples are collected (i.e., every instance of one category is independent of another instance of that category regardless of other categories)  

As we can see we can only carry out parametric statistical anlysis for equal means only in case of year 2022 and 2023 between their respective experience levels.
This is due to the fact the representative sample size for earlier years is very low and highly skewed and requires non-paramtric analysis which is less reliable than parametric.
Hence carrying out Parametrice analysis
we use levene's test to determine the hypothesis of equal variances and then do the one way anova for hypothesis of equal means
we get 

For 2020
- Non-Parametric Kruskal-wallis test p = 0.0016

For 2021
- Non-Parametric Kruskal-wallis test p = 2.8e-12


For 2022
- levenes test
  - p value = 0.13823, indicating equality of variances
- anova one way 
  -  p value = 2.6223e-77, indicating inequality of means

For 2023
- levenes test
  - p value = 0.0471, indicating equality of variances
- anova one way 
  -  p value = 1.24e-42, indicating inequality of means  

**Therefore we conclude that All the Means across their experience levels are statistically different**
'''

for y in ['2022','2023']:
    levene(
        sal.loc[idx[y,exp_levels[0]],'salary_in_usd'].to_numpy(),
        sal.loc[idx[y,exp_levels[1]],'salary_in_usd'].to_numpy(),
        sal.loc[idx[y,exp_levels[2]],'salary_in_usd'].to_numpy(),
        sal.loc[idx[y,exp_levels[3]],'salary_in_usd'].to_numpy()
        )[1]


sm.stats.anova_oneway((
    sal.loc[idx['2022',exp_levels[0]],'salary_in_usd'].to_numpy(),
    sal.loc[idx['2022',exp_levels[1]],'salary_in_usd'].to_numpy(),
    sal.loc[idx['2022',exp_levels[2]],'salary_in_usd'].to_numpy(),
    sal.loc[idx['2022',exp_levels[3]],'salary_in_usd'].to_numpy()
    ),use_var="equal")[1]


sm.stats.anova_oneway((
    sal.loc[idx['2023',exp_levels[0]],'salary_in_usd'].to_numpy(),
    sal.loc[idx['2023',exp_levels[1]],'salary_in_usd'].to_numpy(),
    sal.loc[idx['2023',exp_levels[2]],'salary_in_usd'].to_numpy(),
    sal.loc[idx['2023',exp_levels[3]],'salary_in_usd'].to_numpy()
    ),use_var="unequal")[1]


stats.kruskal(
    sal.loc[idx['2020',exp_levels[0]],'salary_in_usd'].to_numpy(),
    sal.loc[idx['2020',exp_levels[1]],'salary_in_usd'].to_numpy(),
    sal.loc[idx['2020',exp_levels[2]],'salary_in_usd'].to_numpy(),
    sal.loc[idx['2020',exp_levels[3]],'salary_in_usd'].to_numpy()
)


"""
## Employment Type
"""

sal = salaries.copy()
sal.set_index(['work_year','employment_type'],inplace=True)
sal.sort_index(inplace=True)

emp_type_mean = make_subplots(rows=4,cols=4)
for i,year_emp in enumerate(sal.index.unique()):
    stan = StandardScaler().fit_transform(sal.loc[year_emp,'salary_in_usd'].to_numpy().reshape(-1,1)).ravel()
    data = sm.qqplot(stan,line='s').gca().lines
    plt.close()
    emp_type_mean.add_trace(go.Scatter(x=data[0].get_xdata(),y=data[0].get_ydata(),mode="markers"),row=(i//4)+1,col=(i%4)+1)
    emp_type_mean.add_trace(go.Scatter(x=data[1].get_xdata(),y=data[1].get_ydata(),mode="lines"),row=(i//4)+1,col=(i%4)+1)
    emp_type_mean.add_annotation(text=f"{year_emp}",font=dict(size=15),x=0.3,y=1.05,xref="x domain",yref='y domain',xanchor='left',yanchor='bottom',row=(i//4)+1,col=(i%4)+1,bordercolor="white",showarrow=False)
    emp_type_mean.add_annotation(text=f"count :{len(data[0].get_xdata())}",font=dict(size=15),x=0.8,y=1.05,xref="x domain",yref='y domain',xanchor='left',yanchor='bottom',row=(i//4)+1,col=(i%4)+1,bordercolor="white",showarrow=False)

emp_type_mean.update_layout(height=800,showlegend=False,hovermode=False)
emp_type_mean.show()


exp_type_mean_inf = '''
As we can observe the disparity between the respective same level values is high due to the lower count, which make the analysis non parametric   
we can still do welschs anova which will be less reliable than the normal anova
so we rest the case that year level inferences between employment types is avoided
'''

register_page(__name__,order=3,path='/pages/Statistical_analysis',name="Statistial Analysis",title="Statistical Analysis")


options = ['Message','Mean Salary Analysis','Experience Level Mean Analysis']


layout = html.Div([
    html.H1("Statistical Analysis",style={'font-face':'Fira Code'}),
    dcc.Dropdown(
        id='drop1',
        options=options,
        value='Message',    
        searchable=True,
        placeholder="Select options...",
        style={'font-family':'Fira Code'}
    ),
    dcc.Loading(id="Loading1",children=[
        html.Div(id="graphing"),
        dcc.Markdown(id='msg1',style={'font-family':'Fira Code'})
        ],type="cube")
])


@callback(
    [Output('graphing','children'),Output('msg1','children')],
    [Input('drop1','value')]
)
def update_graph(drop1):
    if drop1 == options[0]:
        time.sleep(2)
        return None,msgs
    elif drop1 == options[1]:
        time.sleep(2)
        return dcc.Graph(figure=Mean_Analysis),mean_inf
    elif drop1 == options[2]:
        time.sleep(2)
        return dcc.Graph(figure=exp_level_mean),exp_level_mean_inf

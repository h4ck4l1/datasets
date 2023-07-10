import time
import pycountry as pyc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
pio.templates['mod'] = go.layout.Template(layout=dict(font=dict(family="Fira Code")))
pio.templates.default = "plotly_dark+mod"
from dash import html,dcc,callback,Input,Output,dash_table,register_page

sal = pd.read_csv('https://github.com/h4ck4l1/datasets/raw/main/Normal_practice/ds_salaries.csv')
salaries = sal.copy()

alpha2_names = sal.company_location.unique()
all_counts = {country.alpha_2:country.name for country in pyc.countries}
alpha_dict = {}
for country in alpha2_names:
    alpha_dict[country] = pyc.countries.get(alpha_2=country).name

alpha2_names_1 = sal.employee_residence.unique()
all_counts = {country.alpha_2:country.name for country in pyc.countries}
alpha_dict_1 = {}
for country in alpha2_names_1:
    alpha_dict_1[country] = pyc.countries.get(alpha_2=country).name


sal.drop(['job_title','salary','salary_currency'],axis=1,inplace=True)
sal['work_year'] = sal.work_year.astype(str)
sal['experience_level'] = sal.experience_level.map({'SE':'Senior','MI':'MidInter','EX':'Executive','EN':'Entry'})
sal['employment_type'] = sal.employment_type.map({'FT':'FullTime','CT':'Contract','FL':'FreeLancer','PT':'PartTime'})
sal['company_location'] = sal.company_location.map(alpha_dict)
sal['employee_residence'] = sal.employee_residence.map(alpha_dict_1)
sal_by_exp = sal.groupby(['work_year','experience_level']).aggregate({'salary_in_usd':'mean'}).salary_in_usd
sal_by_emp = sal.groupby(['work_year','employment_type']).aggregate({'salary_in_usd':'mean'}).salary_in_usd
exp_list = ['Entry','MidInter','Senior','Executive']
emp_list = ['FullTime','Contract','FreeLancer','PartTime']
years = ['2020','2021','2022','2023']

salaries['experience_level'] = salaries.experience_level.map({'SE':'Senior','MI':'MidInter','EX':'Executive','EN':'Entry'})
salaries['employment_type'] = salaries.employment_type.map({'FT':'FullTime','CT':'Contract','FL':'FreeLancer','PT':'PartTime'})
salaries['company_location'] = salaries.company_location.map(alpha_dict)
salaries['employee_residence'] = salaries.employee_residence.map(alpha_dict_1)

idx = pd.IndexSlice

sal_by_exp_mean = sal.groupby('work_year').aggregate({'salary_in_usd':'mean'})
sal_by_exp_mean['shifted_salary'] = sal_by_exp_mean.shift(1).fillna(0).loc[:,'salary_in_usd']
sal_by_exp_mean['net_minus'] = sal_by_exp_mean['salary_in_usd'] - sal_by_exp_mean['shifted_salary']
sal_by_exp_mean.loc['2020','shifted_salary'] = 1
sal_by_exp_mean['change'] = (sal_by_exp_mean.net_minus/sal_by_exp_mean.shifted_salary)*100
sal_by_exp_mean.loc['2020','change'] = 0

new_col_names = ['Work Year','Experience Level','Employment Type','Job Title','Salary','Salary Currency','Salary in USD','Employee Residence','Remote ratio','Company Location','Company Size']
new_col_dict = {}
for i in range(len(new_col_names)):
    new_col_dict[salaries.columns[i]] = new_col_names[i]


col_names = ['Work Year','Experience Level','Employment Type','Salary in USD','Employee Residence','Remote ratio','Company Location','Company Size']


col_dict = {}
for i in range(len(col_names)):
    col_dict[sal.columns[i]] = col_names[i]

len(sal)

"""
# Dashtable
"""

tab = dash_table.DataTable(
id='Table',
data=salaries.to_dict('records'),
columns=[{'name':new_col_dict[i],'id':i,'deletable':True,'selectable':True} for i in salaries.columns],
page_size=10,
filter_action='native',
sort_action='native',
row_selectable="multi")

"""
## Experience YoY
"""

exp_yoy = go.Figure()
for exp in exp_list:
    exp_yoy.add_trace(go.Bar(x=years,y=sal_by_exp.loc[idx[:,exp]],name=f"{exp}",text=f"{exp}",textfont=dict(size=20),texttemplate="%{text}<br>%{y:.2s}"))
exp_yoy.update_layout(barmode='stack',height=600)
exp_yoy.update_xaxes(tickfont=dict(size=18),title=dict(text="Work Year",font=dict(size=25)))
exp_yoy.update_yaxes(tickfont=dict(size=18),title=dict(text="Mean in USD",font=dict(size=25)))
exp_yoy.add_annotation(text="Experiece Level Salary in USD YoY",showarrow=False,font=dict(size=30),xanchor="center",yanchor="bottom",xref='x domain',yref='y domain',y=1.05,bordercolor="white",borderpad=10)


"""
#### Experience Inference
"""

Experience_inference = "Graph goes up and up good!"

"""
## Employment YoY
"""

emp_yoy = go.Figure()
for emp in emp_list:
    emp_yoy.add_trace(go.Bar(x=years,y=sal_by_emp.loc[idx[:,emp]],name=f"{emp}",text=f"{emp}",textfont=dict(size=20),texttemplate="%{text}<br>%{y:.2s}"))
emp_yoy.update_layout(barmode='stack',height=650)
emp_yoy.update_xaxes(tickfont=dict(size=18),title=dict(text="Work Year",font=dict(size=25)))
emp_yoy.update_yaxes(tickfont=dict(size=18),title=dict(text="Mean by Type in USD",font=dict(size=25)))
emp_yoy.add_annotation(text="Employment Type Salary in USD YoY",showarrow=False,font=dict(size=30),xanchor="center",yanchor="bottom",xref='x domain',yref='y domain',y=1.05,bordercolor="white")

"""
#### Employment Inference
"""

Employment_inference = "Employment Type mean shots up at 2021 for contract"

"""
## Reduced Mean YoY
"""

mean_yoy = go.Figure()
mean_yoy.add_annotation(text="Reduced Mean with %Change wrt to previous year",showarrow=False,font=dict(size=30),xanchor="center",yanchor="bottom",xref='x domain',yref='y domain',y=1.05,bordercolor="white",borderpad=10)
mean_yoy.add_trace(go.Scatter(x=years,y=sal.groupby(['work_year']).aggregate({'salary_in_usd':'mean'}),mode='lines+markers+text',line=dict(dash='dashdot',color="springgreen"),marker=dict(size=50,opacity=0.3),name='Mean YoY',textposition="top center",textfont=dict(size=15)))
mean_yoy.add_trace(go.Bar(x=years,y=sal.groupby(['work_year']).aggregate({'salary_in_usd':'mean'}),marker=dict(color="royalblue"),name="Mean YoY",text=sal_by_exp_mean.change,texttemplate="<br>\n<br>\n<br>\n<b><i>Mean :%{y:.2s}<br>%Change:%{text:.3s}</i></b>",textposition="inside",textfont=dict(size=20,color="rgb(225,145,65)")))
mean_yoy.update_traces(width=0.7,selector=dict(type="bar"))
mean_yoy.update_layout(height=700,width=1200)
mean_yoy.update_xaxes(showline=False,griddash="dash",gridcolor="rgba(255,255,255,0.2)",tickfont=dict(size=20),title="Work Year",titlefont=dict(size=20))
mean_yoy.update_yaxes(range=[0,2e5],griddash="dash",gridcolor="rgba(255,255,255,0.2)",tickfont=dict(size=20),title="Mean reduced YoY",titlefont=dict(size=20))

"""
#### Mean Inference
"""

Mean_inference = "There was a good jump from 2021 to 2022 which can mean the employment along with payscale gradually increased, but it can also be due to growing interest towards datascience field ingeneral"

"""
## Remote Ratio
"""

sal_by_rem = pd.pivot_table(sal,index=['work_year'],columns=['remote_ratio'],values=['salary_in_usd'],aggfunc=['count','mean'])

remote_pie = go.Figure()
remote_pie.add_trace(go.Pie(values=sal_by_rem.loc['2020',idx['count','salary_in_usd']],labels=[0,50,100],texttemplate="<b>Remote Ratio</b>:<b>%{label}</b><br>count:%{value}<br>%{text:.2s}</br>",textfont=dict(size=20),sort=False,direction="clockwise",textposition="outside",text=sal_by_rem.loc['2020',idx['mean','salary_in_usd']]))
frames = []
steps = []
for year in years:
    frames.append(go.Frame(data=[go.Pie(values=sal_by_rem.loc[year,idx['count','salary_in_usd']],labels=[0,50,100],texttemplate="<b>Remote Ratio</b>:<b>%{label}</b><br>count:%{value}<br>%{text:.2s}</br>",textfont=dict(size=20),sort=False,direction="clockwise",textposition="outside",text=sal_by_rem.loc[year,idx['mean','salary_in_usd']])],layout=dict(autosize=False),name=year))
    steps.append(dict(
        args=[[year],dict(frame=dict(duration=1500,redraw=True),mode="immediate",transition=dict(duration=1500,easing="quadratic-in-out"))],
        label=year,
        method="animate"
    ))
remote_pie.update(frames=frames)
remote_pie.update_layout(
    updatemenus=[dict(
        buttons=[dict(label="Play",method="animate",args=[None,dict(frame=dict(duration=1500,redraw=True),fromcurrent=True,transition=dict(duration=1500,easing="circle-in-out"))])],
        xanchor="left",
        yanchor="top",
        x=0.1,
        y=0,
        showactive=False,
        type="buttons"
    )],
    sliders=[dict(
        steps=steps,
        xanchor="left",
        yanchor="top",
        x=0,
        y=-0.09,
        len=0.9,
        active=0,
        currentvalue=dict(
            font=dict(size=20),
            prefix="Year :",
            xanchor="right",
            visible=True
        ),
        transition=dict(duration=1500,easing="cubic-in-out"),
        pad=dict(r=10,b=10),
        font=dict(size=20)
    )]
)
remote_pie.update_layout(autosize=False,height=600,width=1100,margin=dict(t=5,r=10),legend=dict(font=dict(size=20),xanchor="right",yanchor="top",x=0,y=0.05))

"""
#### Remote ratio inference
"""

Remote_ratio = "As it can be seen that people who worked remotely were higher in 2020 and eventually due to Covid reasons and companies started to reopen and employees gradually started working in office again"

"""
## Company Location
"""

sal_by_comp = sal.groupby(['company_location']).aggregate({'salary_in_usd':'mean'})
q_25 = sal_by_comp.salary_in_usd.quantile(0.25)
q_75 = sal_by_comp.salary_in_usd.quantile(0.75)

q_25_countries = sal_by_comp.loc[sal_by_comp['salary_in_usd'] < q_25].index
q_75_countries = sal_by_comp.loc[sal_by_comp['salary_in_usd'] > q_75].index
sal['q'] = np.NaN
sal.loc[sal['company_location'].isin(q_25_countries),'q'] = 1
sal.loc[sal['company_location'].isin(q_75_countries),'q'] = 3
sal.fillna(2,inplace=True)

company_loc = make_subplots(rows=3,cols=1,row_heights=[0.25,0.5,0.25],vertical_spacing=0.03,row_titles=['Companies below First Quantile','Companies in between First and Third Quantile','Companies in Third Quatile'])
company_loc.add_trace(go.Histogram(x=sal.loc[sal.q==1,'salary_in_usd'],y=sal.loc[sal.q==1,'company_location'],name="First Quantile",orientation='h',histfunc='avg'),row=1,col=1)
company_loc.add_trace(go.Histogram(x=sal.loc[sal.q==2,'salary_in_usd'],y=sal.loc[sal.q==2,'company_location'],name="Between First and Third Quantile",orientation='h',histfunc='avg'),row=2,col=1)
company_loc.add_trace(go.Histogram(x=sal.loc[sal.q==3,'salary_in_usd'],y=sal.loc[sal.q==3,'company_location'],name="Third Quantile",orientation='h',histfunc='avg'),row=3,col=1)
company_loc.update_yaxes(categoryorder="total descending",title="Country where company is located",title_font_size=20)
company_loc.update_xaxes(title="Mean salary in USD",title_font_size=20)
company_loc.update_annotations(font=dict(size=30))
company_loc.update_layout(height=2500)

"""
#### Company Location Inference
"""

pd.crosstab(index=sal.company_location,columns=sal.work_year).loc["United States"]

Company_inf = '''
The Mean salary of Israel is highest followed by Puero Rico and United States, though the count of United states is far higher than rest, so most of the mean salaries can be highly skewed
'''

register_page(__name__,order=2,path='/pages/sal',name="Visual Presentation",title="Visual Presentation")

layout = html.Div([
    html.Br(),
    dcc.Dropdown(
        id='drop',
        options=['Table','Experience YoY','Employment YoY','Mean reduced YoY','Remote Ratio YoY','Company Location'],
        value='Table',
        style={'backgroundColor':'black','color':'white','font-family':'Fira Code'}
    ),
    dcc.Loading(id="loading",children=[
        html.Div(id='Table_or_graphs'),
        html.Br(),
        html.Br(),
        html.H2(id='Inf',style={'font-family':'Fira Code'}),
        dcc.Markdown(id='msg',style={'white-space':'pre','font-family':'Fira Code'})
    ],type='cube')
])

@callback(
    [
        Output('Table_or_graphs','children'),
        Output('Inf','children'),
        Output('msg','children')
        ],
    [Input('drop','value')]
)
def update_graph(value):

    if value == 'Table':
        time.sleep(2)
        return tab,None,None
    elif value == 'Experience YoY':
        time.sleep(2)
        return dcc.Graph(figure=exp_yoy),"Inferences :",Experience_inference
    elif value == 'Employment YoY':
        time.sleep(2)
        return dcc.Graph(figure=emp_yoy),"Inferences :",Employment_inference
    elif value == 'Mean reduced YoY':
        time.sleep(2)
        return dcc.Graph(figure=mean_yoy),"Inferences :",Mean_inference
    elif value == 'Remote Ratio YoY':
        time.sleep(2)
        return dcc.Graph(figure=remote_pie),"Inferences :",Remote_ratio
    elif value == 'Company Location':
        time.sleep(2)
        return dcc.Graph(figure=company_loc),"Inferences :",Company_inf

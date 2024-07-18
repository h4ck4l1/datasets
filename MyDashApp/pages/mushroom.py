import os,sys,time,math,re
from glob import glob
import numpy as np
import pandas as pd
import polars as pl
import joblib
from typing import Literal, Optional
import duckdb
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
import catboost as cgb
import xgboost as xgb
import polars.selectors as cs
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates["mod"] = go.layout.Template(
    layout=dict(
        font=dict(
            family="Monaco",
            size=15
        ),
        xaxis=dict(
            showgrid=False
        ),
        yaxis=dict(
            showgrid=False
        ),
        legend=dict(
            title=dict(
                font=dict(
                    color="cyan"
                )
            )
        )
    )
)
pio.templates.default = "mod+plotly_dark"
from fitter import Fitter
from dash import Dash,html, dcc, Input, Output,register_page,callback,ctx
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc

class Download:

    from zipfile import ZipFile
    import os,shutil,json,subprocess


    def __new__(cls,*args,**kwargs):
        kaggle_dir = cls.os.path.expanduser("~/.kaggle")
        cls.os.makedirs(kaggle_dir,exist_ok=True)
        with open(cls.os.path.join(kaggle_dir,"kaggle.json"),"w") as f_out:
            cls.json.dump(
                {
                    "username":"sohailmd123",
                    "key":"6aa4d9ead00c5ec7d3bcdadb2771d902"
                },
                f_out
            )
        cls.os.chmod(cls.os.path.join(kaggle_dir,"kaggle.json"),0o600)

        return super(Download,cls).__new__(cls)

    def __init__(self,path=None,competition_name=None):
        self.competition_name = competition_name
        self.dirpath = "." or path

    def download_competition_file(self,filename:str):
        self.subprocess.run(f"kaggle competitions download {self.competition_name} -f {filename}",shell=True)
        with self.ZipFile(f"{filename}.zip","r") as f_out:
            f_out.extractall("{self.dirpath}")
        self.os.remove(f"{filename}.zip")
        return " ".join(self.os.listdir())

    def download_all_competition_files(self):
        self.subprocess.run(f"kaggle competitions download -c {self.competition_name}",shell=True)
        with self.ZipFile(f"{self.competition_name}.zip","r") as f_out:
            f_out.extractall("{self.dirpath}")
        self.os.remove(f"{self.competition_name}.zip")
        return " ".join(self.os.listdir())

    def download_dataset(self,api_command:str):
        self.subprocess.run(api_command,shell=True)
        dataset_name = api_command.split("/")[-1]
        with self.ZipFile(f"{dataset_name}.zip","r") as f_out:
            f_out.extractall(f"{self.dirpath}/{dataset_name}/")
        self.os.remove(f"{dataset_name}.zip")
        return " ".join(self.os.listdir())


    def upload_dataset(self,title:str= 'Differet fits for Dataset', id: str = 'sohailmd123/eda-saved', license_name: str = "CC0-1.0", message: str = "Updated dataset"):
        metadata = {
            "title": title,
            "id": id,
            "licenses": [
                {
                    "name": license_name
                }
            ]
        }
        folder_path = id.split("/")[-1]
        if not self.os.path.exists(f"{folder_path}"):
            self.os.makedirs(f"{folder_path}")
            return f"Folder {folder_path} was created"
        with open(f"{folder_path}/dataset-metadata.json", "w") as f_out:
            self.json.dump(metadata, f_out)
            print("wrote metadata")
        result = self.subprocess.run(f"kaggle datasets status {id}", shell=True, capture_output=True, text=True)
        if "403 - Forbidden" in result.stdout:
            result = self.subprocess.run(f"kaggle datasets create -p {folder_path}", shell=True, capture_output=True,text=True)
        else:
            result = self.subprocess.run(f"kaggle datasets version -p {folder_path} -m \"{message}\"", shell=True,capture_output=True,text=True)
        return f"Dataset {title} uploaded/updated on Kaggle\n{result.stdout}"

    def download_uploadable_dataset(self,api_command:str,title:str="Different fits for Dataset",license_name:str="CC0-1.0"):
        self.subprocess.run(api_command,shell=True)
        dataset_name = api_command.split("/")[-1]
        with self.ZipFile(f"{dataset_name}.zip","r") as f_out:
            f_out.extractall(f"{self.dirpath}/{dataset_name}/")
        self.os.remove(f"{dataset_name}.zip")
        id = api_command.split(" ")[-1]
        metadata = {
            "title": title,
            "id": id,
            "licenses": [
                {
                    "name": license_name
                }
            ]
        }
        folder_name = id.split("/")[-1]
        with open(f"{folder_name}/dataset-metadata.json","w") as f_out:
            self.json.dump(metadata,f_out)
            print("wrote metadata")
        return " ".join(self.os.listdir())

download = Download()

download.download_uploadable_dataset("kaggle datasets download -d sohailmd123/eda-saved")

def get_image(name:str):
    all_images = glob("assets/*.png") + \
                 glob("assets/*.jpg") + \
                 glob("assets/*.jpeg") + \
                 glob("assets/*.svg")
    for img in all_images:
        img_name = os.path.basename(img).split(".")[0]
        if name in img_name:
            return img
        


def get_stat(col:str,stat:Literal["mean","std","median","mode"]):
    if stat == "mean":
        return df.select(col).mean().item()
    elif stat == "std":
        return df.select(col).std().item()
    elif stat == "median":
        return df.select(col).median().item()
    else:
        return (
            df
            .select(
                pl.col(col).value_counts(sort=True)
            )
            .item(0,0)
            [col]
        )


def create_fit_and_upload(df:pl.DataFrame,dataset_name:str,col_name:str,folder_name:str="eda-saved",bins:int=None,refit:bool=False,plot_it:bool=False,return_output:bool=False):
    file_name = dataset_name + "_" + col_name.replace(" ","_")
    if not refit:
        for f in glob(f"{folder_name}/*.pkl"):
            if file_name in os.path.basename(f):
                print("found file",f)
                fitr = joblib.load(f)
                return fitr,(
                                pl.from_pandas(fitr.summary(10,plot=plot_it).reset_index(names=['distribution']))
                                .select(pl.col('distribution'),pl.col('ks_pvalue').map_elements(lambda x: float(f'{x:.3e}'),return_dtype=float))
                            )
    if bins:
        fitr = Fitter(df.select(col_name),bins=bins,timeout=300)
    else:
        fitr = Fitter(df.select(col_name),bins=df.select(col_name).unique().height,timeout=300)
    fitr.fit(progress=True)
    joblib.dump(fitr,f"{folder_name}/{file_name}.pkl")
    if not return_output:
        download.upload_dataset()
    else:
        print(download.upload_dataset())
    return fitr,(
                    pl.from_pandas(fitr.summary(10,plot=plot_it).reset_index(names=['distribution']))
                    .select(pl.col('distribution'),pl.col('ks_pvalue').map_elements(lambda x: float(f'{x:.3e}'),return_dtype=float))
                )

def get_histogram(df:pl.DataFrame,col_name:str,x:float=0.5,n_bins:int=None,bar_gap:float=0):
    if not n_bins:
        n_bins = df[col_name].unique().shape[0]
    counts,binsx = np.histogram(df[col_name],bins=n_bins,density=True)
    binsx = [(binsx[i+1]+_)/2.0 for i,_ in enumerate(binsx[:-1])]
    counts *= 10e4
    fig = go.Figure()
    fig.add_traces(go.Bar(x=binsx,y=counts,marker=dict(line=dict(color="#636efa",width=0)),showlegend=False))
    fig.update_layout(bargap=bar_gap)
    if n_bins > 100:
        fig.add_trace(go.Scatter(x=np.full(int(np.max(counts)),get_stat(col_name,"mean")),hovertemplate="Mean: %{x}",name=f"Mean of {col_name}",mode="lines",line=dict(dash="dashdot",color="aqua"),showlegend=False))
        fig.add_trace(go.Scatter(x=np.full(int(np.max(counts)),get_stat(col_name,"median")),hovertemplate="Median: %{x}",name=f"Median of {col_name}",mode="lines",line=dict(dash="dashdot",color="aqua"),showlegend=False))
        fig.add_trace(go.Scatter(x=np.full(int(np.max(counts)),get_stat(col_name,"mode")),hovertemplate="Mode: %{x}",name=f"Mode of {col_name}",mode="lines",line=dict(dash="dashdot",color="aqua"),showlegend=False))
        fig.add_annotation(x=get_stat(col_name,"mean"),y=-0.08,yref="paper",showarrow=False,textangle=-45,text="Mean",bordercolor="white")
        fig.add_annotation(x=get_stat(col_name,"median"),y=-0.08,yref="paper",showarrow=False,textangle=-45,text="Median",bordercolor="white")
        fig.add_annotation(x=get_stat(col_name,"mode"),y=-0.08,yref="paper",showarrow=False,textangle=-45,text="Mode",bordercolor="white")
    else:
        fig.add_annotation(
            text=f"Mean: {get_stat(col_name,'mean'):.2f}<br>Median: {get_stat(col_name,'median')}<br>Mode: {get_stat(col_name,'mode')}",
            x=x,
            y=0.9,
            xref="paper",
            yref="paper",
            font=dict(family="Monaco",size=21),
            bordercolor="white",
            borderwidth=1,
            borderpad=10,
        )
        fig.update_xaxes(tickvals=binsx,ticktext=[_ for _ in range(len(binsx))],ticks="outside",tickson="boundaries")
    return fig


con = duckdb.connect()

df = (
    con.query(
        '''
        select * from read_csv('DataFolder/mushroom_cleaned.csv')
        '''
    )
    .pl()
    .select(pl.all().shrink_dtype())
    .pipe(lambda df: df.rename({col:col.replace('-',' ') for col in df.columns}))
)

cat_cols = sorted(cs.expand_selector(df,(cs.integer() | cs.boolean() | cs.string() | cs.categorical())))
cont_cols = sorted(cs.expand_selector(df,(cs.float())))

cols_list = cat_cols + cont_cols
menu_options_list = []

menu_options_list.append(
    html.Li(
        html.Span(
            [
                dcc.Link("INDEX",href="/mushroom",className="subpage-link")
            ]
        )
    )
)

for col in cols_list:
    menu_options_list.append(
        html.Li(
            html.Span(
                [
                    dcc.Link(
                        f"MUSHROOM {col.upper()}",
                        href=f"/mushroom/mushroom_{col.replace(' ','_')}",
                        className="subpage-link"
                    )
                ]
            )
        )
    )

mushroom_menu = html.Div(
    [
        html.Ul(
            html.Li(
                html.Span(
                    [
                        html.H1("MUSHROOM EDA",className="subpage-dropdown-button-title"),
                        html.Img(src=get_image("mushroom_index"),className="subpage-dropdown-button-icon")
                    ]
                )
            ),
            className="subpage-dropdown-button"
        ),
        html.Div(html.Span(className="arrow"),className="arrow-space"),
        html.Ul(menu_options_list,className="subpage-dropdown-box"
        )
    ],
    className="subpage-total-dropdown"
)


mushroom_msg = '''

This dataset is a cleaned version of the original Mushroom Dataset for Binary Classification Available at UCI Library.
This dataset was cleaned using various techniques such as Modal imputation, one-hot encoding, z-score normalization, and feature selection.
It contains 9 columns:

- Cap Diameter
- Cap Shape
- Gill Attachment
- Gill Color
- Stem Height
- Stem Width
- Stem Color
- Season
- Target Class - Is it edible or not?
- The Target Class contains two values - 0 or 1 - where 0 refers to edible and 1 refers to poisonous.

'''
mushroom_index = html.Div(
    [
        mushroom_menu,
        html.Div(html.H1("mushroom index",className="page-heading"),className="heading-divs"),
        dcc.Markdown(mushroom_msg,className="subpage-markdown")
    ],
    className="subpage-content"
)

register_page(
    "mushroom_index",
    path="/mushroom",
    layout=mushroom_index
)


mushroom_first_cat_col_fitter,mushroom_first_cat_col_df = create_fit_and_upload(df,"mushroom",cat_cols[0],bins=1000)
mushroom_first_cat_col_layout = html.Div(
    [
        mushroom_menu,
        html.Div(html.H1(f"MUSHROOM {cat_cols[0].upper()}",className="page-heading"),className="heading-divs"),
        DataTable(
            data=mushroom_first_cat_col_df.to_dicts(),
            columns=[{'name':i,'id':i} for i in mushroom_first_cat_col_df.columns],
            style_header={'textAlign':'right','text-transform':'uppercase','width':'20vw'},
            style_cell={'textAlign':'right','text-transform':'uppercase','backgroundColor':'#4c3bcf'},
            style_data={'backgroundColor':'#4c3bcf'},
            style_table={'width':'30vw','left':'35vw','padding':'50px'},
            id='mushroom_first_cat_col_table',
            style_cell_conditional=[
                {
                    'if':{'column_id':"distribution"},
                    'textAlign':'left'
                }
            ],
            style_header_conditional=[
                {
                    'if': {'column_id':'distribution'},
                    'textAlign':'left'
                }
            ],
            style_data_conditional=[
                {
                    'if': {'state':'active'},
                    'backgroundColor':'#4b70f5'
                }
            ]
        ),
        dcc.Graph(id="mushroom_first_cat_col_hist",className="graph-content")
    ],
    className="subpage-content"
)

@callback(Output('mushroom_first_cat_col_hist','figure'),Input('mushroom_first_cat_col_table','active_cell'))
def update_mushroom_first_cat_col_fig(active_cell:dict):
    if active_cell:
        row = active_cell.get('row',0)
    else:
        row = 0
    active_cell = mushroom_first_cat_col_df.slice(row,1).to_dicts()[0]
    fig = get_histogram(df,cat_cols[0],n_bins=1000)
    fig.add_trace(
        go.Scatter(
            x=mushroom_first_cat_col_fitter.x,
            y=mushroom_first_cat_col_fitter.fitted_pdf[active_cell.get('distribution')] * 10e4,
            text=[str(active_cell.get('ks_pvalue')) for _ in range(len(mushroom_first_cat_col_fitter.x))],
            texttemplate="(%{x}, %{y})<br>p_value: %{text}",
            name=active_cell.get('distribution')
        )
    )
    return fig


register_page(
    f"mushroom {cat_cols[0]}",
    path=f"/mushroom/mushroom_{cat_cols[0].replace(' ','_')}",
    layout=mushroom_first_cat_col_layout
)


mushroom_second_cat_col_fitter,mushroom_second_cat_col_df = create_fit_and_upload(df,"mushroom",cat_cols[1])
mushroom_second_cat_col_layout = html.Div(
    [
        mushroom_menu,
        html.Div(html.H1(f"MUSHROOM {cat_cols[1].upper()}",className="page-heading"),className="heading-divs"),
        DataTable(
            data=mushroom_second_cat_col_df.to_dicts(),
            columns=[{'name':i,'id':i} for i in mushroom_second_cat_col_df.columns],
            style_header={'textAlign':'right','text-transform':'uppercase','width':'20vw'},
            style_cell={'textAlign':'right','text-transform':'uppercase','backgroundColor':'#4c3bcf'},
            style_data={'backgroundColor':'#4c3bcf'},
            style_table={'width':'30vw','left':'35vw','padding':'50px'},
            id='mushroom_second_cat_col_table',
            style_cell_conditional=[
                {
                    'if':{'column_id':"distribution"},
                    'textAlign':'left'
                }
            ],
            style_header_conditional=[
                {
                    'if': {'column_id':'distribution'},
                    'textAlign':'left'
                }
            ],
            style_data_conditional=[
                {
                    'if': {'state':'active'},
                    'backgroundColor':'#4b70f5'
                }
            ]
        ),
        dcc.Graph(id="mushroom_second_cat_col_hist",className="graph-content")
    ],
    className="subpage-content"
)



@callback(Output('mushroom_second_cat_col_hist','figure'),Input('mushroom_second_cat_col_table','active_cell'))
def update_mushroom_second_cat_col_fig(active_cell:dict):
    if active_cell:
        row = active_cell.get('row',0)
    else:
        row = 0
    active_cell = mushroom_second_cat_col_df.slice(row,1).to_dicts()[0]
    fig = get_histogram(df,cat_cols[1],bar_gap=0.2)
    fig.add_trace(
        go.Scatter(
            x=mushroom_second_cat_col_fitter.x,
            y=mushroom_second_cat_col_fitter.fitted_pdf[active_cell.get('distribution')] * 10e4,
            text=[str(active_cell.get('ks_pvalue')) for _ in range(len(mushroom_second_cat_col_fitter.x))],
            texttemplate="(%{x}, %{y})<br>p_value: %{text}",
            name=active_cell.get('distribution')
        )
    )
    return fig


register_page(
    f"mushroom {cat_cols[1]}",
    path=f"/mushroom/mushroom_{cat_cols[1].replace(' ','_')}",
    layout=mushroom_second_cat_col_layout
)


mushroom_third_cat_col_fitter,mushroom_third_cat_col_df = create_fit_and_upload(df,"mushroom",cat_cols[2])
mushroom_third_cat_col_layout = html.Div(
    [
        mushroom_menu,
        html.Div(html.H1(f"MUSHROOM {cat_cols[2].upper()}",className="page-heading"),className="heading-divs"),
        DataTable(
            data=mushroom_third_cat_col_df.to_dicts(),
            columns=[{'name':i,'id':i} for i in mushroom_third_cat_col_df.columns],
            style_header={'textAlign':'right','text-transform':'uppercase','width':'20vw'},
            style_cell={'textAlign':'right','text-transform':'uppercase','backgroundColor':'#4c3bcf'},
            style_data={'backgroundColor':'#4c3bcf'},
            style_table={'width':'30vw','left':'35vw','padding':'50px'},
            id='mushroom_third_cat_col_table',
            style_cell_conditional=[
                {
                    'if':{'column_id':"distribution"},
                    'textAlign':'left'
                }
            ],
            style_header_conditional=[
                {
                    'if': {'column_id':'distribution'},
                    'textAlign':'left'
                }
            ],
            style_data_conditional=[
                {
                    'if': {'state':'active'},
                    'backgroundColor':'#4b70f5'
                }
            ]
        ),
        dcc.Graph(id="mushroom_third_cat_col_hist",className="graph-content")
    ],
    className="subpage-content"
)

@callback(Output('mushroom_third_cat_col_hist','figure'),Input('mushroom_third_cat_col_table','active_cell'))
def update_mushroom_third_cat_col_fig(active_cell:dict):
    if active_cell:
        row = active_cell.get('row',0)
    else:
        row = 0
    active_cell = mushroom_third_cat_col_df.slice(row,1).to_dicts()[0]
    fig = get_histogram(df,cat_cols[2],bar_gap=0.2)
    fig.add_trace(
        go.Scatter(
            x=mushroom_third_cat_col_fitter.x,
            y=mushroom_third_cat_col_fitter.fitted_pdf[active_cell.get('distribution')] * 10e4,
            text=[str(active_cell.get('ks_pvalue')) for _ in range(len(mushroom_third_cat_col_fitter.x))],
            texttemplate="(%{x}, %{y})<br>p_value: %{text}",
            name=active_cell.get('distribution')
        )
    )
    return fig

register_page(
    f"mushroom {cat_cols[2]}",
    path=f"/mushroom/mushroom_{cat_cols[2].replace(' ','_')}",
    layout=mushroom_third_cat_col_layout
)


mushroom_fourth_cat_col_fitter,mushroom_fourth_cat_col_df = create_fit_and_upload(df,"mushroom",cat_cols[3])
mushroom_fourth_cat_col_layout = html.Div(
    [
        mushroom_menu,
        html.Div(html.H1(f"MUSHROOM {cat_cols[3].upper()}",className="page-heading"),className="heading-divs"),
        DataTable(
            data=mushroom_fourth_cat_col_df.to_dicts(),
            columns=[{'name':i,'id':i} for i in mushroom_fourth_cat_col_df.columns],
            style_header={'textAlign':'right','text-transform':'uppercase','width':'20vw'},
            style_cell={'textAlign':'right','text-transform':'uppercase','backgroundColor':'#4c3bcf'},
            style_data={'backgroundColor':'#4c3bcf'},
            style_table={'width':'30vw','left':'35vw','padding':'50px'},
            id='mushroom_fourth_cat_col_table',
            style_cell_conditional=[
                {
                    'if':{'column_id':"distribution"},
                    'textAlign':'left'
                }
            ],
            style_header_conditional=[
                {
                    'if': {'column_id':'distribution'},
                    'textAlign':'left'
                }
            ],
            style_data_conditional=[
                {
                    'if': {'state':'active'},
                    'backgroundColor':'#4b70f5'
                }
            ]
        ),
        dcc.Graph(id="mushroom_fourth_cat_col_hist",className="graph-content")
    ],
    className="subpage-content"
)

@callback(Output('mushroom_fourth_cat_col_hist','figure'),Input('mushroom_fourth_cat_col_table','active_cell'))
def update_mushroom_fourth_cat_col_fig(active_cell:dict):
    if active_cell:
        row = active_cell.get('row',0)
    else:
        row = 0
    active_cell = mushroom_fourth_cat_col_df.slice(row,1).to_dicts()[0]
    fig = get_histogram(df,cat_cols[3],bar_gap=0.2)
    fig.add_trace(
        go.Scatter(
            x=mushroom_fourth_cat_col_fitter.x,
            y=mushroom_fourth_cat_col_fitter.fitted_pdf[active_cell.get('distribution')] * 10e4,
            text=[str(active_cell.get('ks_pvalue')) for _ in range(len(mushroom_fourth_cat_col_fitter.x))],
            texttemplate="(%{x}, %{y})<br>p_value: %{text}",
            name=active_cell.get('distribution')
        )
    )
    return fig

register_page(
    f"mushroom {cat_cols[3]}",
    path=f"/mushroom/mushroom_{cat_cols[3].replace(' ','_')}",
    layout=mushroom_fourth_cat_col_layout
)


mushroom_fifth_cat_col_fitter,mushroom_fifth_cat_col_df = create_fit_and_upload(df,"mushroom",cat_cols[4])
mushroom_fifth_cat_col_layout = html.Div(
    [
        mushroom_menu,
        html.Div(html.H1(f"MUSHROOM {cat_cols[4].upper()}",className="page-heading"),className="heading-divs"),
        DataTable(
            data=mushroom_fifth_cat_col_df.to_dicts(),
            columns=[{'name':i,'id':i} for i in mushroom_fifth_cat_col_df.columns],
            style_header={'textAlign':'right','text-transform':'uppercase','width':'20vw'},
            style_cell={'textAlign':'right','text-transform':'uppercase','backgroundColor':'#4c3bcf'},
            style_data={'backgroundColor':'#4c3bcf'},
            style_table={'width':'30vw','left':'35vw','padding':'50px'},
            id='mushroom_fifth_cat_col_table',
            style_cell_conditional=[
                {
                    'if':{'column_id':"distribution"},
                    'textAlign':'left'
                }
            ],
            style_header_conditional=[
                {
                    'if': {'column_id':'distribution'},
                    'textAlign':'left'
                }
            ],
            style_data_conditional=[
                {
                    'if': {'state':'active'},
                    'backgroundColor':'#4b70f5'
                }
            ]
        ),
        dcc.Graph(id="mushroom_fifth_cat_col_hist",className="graph-content")
    ],
    className="subpage-content"
)

@callback(Output('mushroom_fifth_cat_col_hist','figure'),Input('mushroom_fifth_cat_col_table','active_cell'))
def update_mushroom_fifth_cat_col_fig(active_cell:dict):
    if active_cell:
        row = active_cell.get('row',0)
    else:
        row = 0
    active_cell = mushroom_fifth_cat_col_df.slice(row,1).to_dicts()[0]
    fig = get_histogram(df,cat_cols[4],bar_gap=0.2)
    fig.add_trace(
        go.Scatter(
            x=mushroom_fifth_cat_col_fitter.x,
            y=mushroom_fifth_cat_col_fitter.fitted_pdf[active_cell.get('distribution')] * 10e4,
            text=[str(active_cell.get('ks_pvalue')) for _ in range(len(mushroom_fifth_cat_col_fitter.x))],
            texttemplate="(%{x}, %{y})<br>p_value: %{text}",
            name=active_cell.get('distribution')
        )
    )
    return fig

register_page(
    f"mushroom {cat_cols[4]}",
    path=f"/mushroom/mushroom_{cat_cols[4].replace(' ','_')}",
    layout=mushroom_fifth_cat_col_layout
)



mushroom_sixth_cat_col_fitter,mushroom_sixth_cat_col_df = create_fit_and_upload(df,"mushroom",cat_cols[5])
mushroom_sixth_cat_col_layout = html.Div(
    [
        mushroom_menu,
        html.Div(html.H1(f"MUSHROOM {cat_cols[5].upper()}",className="page-heading"),className="heading-divs"),
        DataTable(
            data=mushroom_sixth_cat_col_df.to_dicts(),
            columns=[{'name':i,'id':i} for i in mushroom_sixth_cat_col_df.columns],
            style_header={'textAlign':'right','text-transform':'uppercase','width':'20vw'},
            style_cell={'textAlign':'right','text-transform':'uppercase','backgroundColor':'#4c3bcf'},
            style_data={'backgroundColor':'#4c3bcf'},
            style_table={'width':'30vw','left':'35vw','padding':'50px'},
            id='mushroom_sixth_cat_col_table',
            style_cell_conditional=[
                {
                    'if':{'column_id':"distribution"},
                    'textAlign':'left'
                }
            ],
            style_header_conditional=[
                {
                    'if': {'column_id':'distribution'},
                    'textAlign':'left'
                }
            ],
            style_data_conditional=[
                {
                    'if': {'state':'active'},
                    'backgroundColor':'#4b70f5'
                }
            ]
        ),
        dcc.Graph(id="mushroom_sixth_cat_col_hist",className="graph-content")
    ],
    className="subpage-content"
)

@callback(Output('mushroom_sixth_cat_col_hist','figure'),Input('mushroom_sixth_cat_col_table','active_cell'))
def update_mushroom_sixth_cat_col_fig(active_cell:dict):
    if active_cell:
        row = active_cell.get('row',0)
    else:
        row = 0
    active_cell = mushroom_sixth_cat_col_df.slice(row,1).to_dicts()[0]
    fig = get_histogram(df,cat_cols[5],bar_gap=0.2)
    fig.add_trace(
        go.Scatter(
            x=mushroom_sixth_cat_col_fitter.x,
            y=mushroom_sixth_cat_col_fitter.fitted_pdf[active_cell.get('distribution')] * 10e4,
            text=[str(active_cell.get('ks_pvalue')) for _ in range(len(mushroom_sixth_cat_col_fitter.x))],
            texttemplate="(%{x}, %{y})<br>p_value: %{text}",
            name=active_cell.get('distribution')
        )
    )
    return fig

register_page(
    f"mushroom {cat_cols[5]}",
    path=f"/mushroom/mushroom_{cat_cols[5].replace(' ','_')}",
    layout=mushroom_sixth_cat_col_layout
)


mushroom_seventh_cat_col_fitter,mushroom_seventh_cat_col_df = create_fit_and_upload(df,"mushroom",cat_cols[6],bins=1000)
mushroom_seventh_cat_col_layout = html.Div(
    [
        mushroom_menu,
        html.Div(html.H1(f"MUSHROOM {cat_cols[6].upper()}",className="page-heading"),className="heading-divs"),
        DataTable(
            data=mushroom_seventh_cat_col_df.to_dicts(),
            columns=[{'name':i,'id':i} for i in mushroom_seventh_cat_col_df.columns],
            style_header={'textAlign':'right','text-transform':'uppercase','width':'20vw'},
            style_cell={'textAlign':'right','text-transform':'uppercase','backgroundColor':'#4c3bcf'},
            style_data={'backgroundColor':'#4c3bcf'},
            style_table={'width':'30vw','left':'35vw','padding':'50px'},
            id='mushroom_seventh_cat_col_table',
            style_cell_conditional=[
                {
                    'if':{'column_id':"distribution"},
                    'textAlign':'left'
                }
            ],
            style_header_conditional=[
                {
                    'if': {'column_id':'distribution'},
                    'textAlign':'left'
                }
            ],
            style_data_conditional=[
                {
                    'if': {'state':'active'},
                    'backgroundColor':'#4b70f5'
                }
            ]
        ),
        dcc.Graph(id="mushroom_seventh_cat_col_hist",className="graph-content")
    ],
    className="subpage-content"
)

@callback(Output('mushroom_seventh_cat_col_hist','figure'),Input('mushroom_seventh_cat_col_table','active_cell'))
def update_mushroom_seventh_cat_col_fig(active_cell:dict):
    if active_cell:
        row = active_cell.get('row',0)
    else:
        row = 0
    active_cell = mushroom_seventh_cat_col_df.slice(row,1).to_dicts()[0]
    fig = get_histogram(df,cat_cols[6],bar_gap=0.2)
    fig.add_trace(
        go.Scatter(
            x=mushroom_seventh_cat_col_fitter.x,
            y=mushroom_seventh_cat_col_fitter.fitted_pdf[active_cell.get('distribution')] * 10e4,
            text=[str(active_cell.get('ks_pvalue')) for _ in range(len(mushroom_seventh_cat_col_fitter.x))],
            texttemplate="(%{x}, %{y})<br>p_value: %{text}",
            name=active_cell.get('distribution')
        )
    )
    return fig

register_page(
    f"mushroom {cat_cols[6]}",
    path=f"/mushroom/mushroom_{cat_cols[6].replace(' ','_')}",
    layout=mushroom_seventh_cat_col_layout
)



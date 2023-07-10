#!/home/sohail/anaconda3/bin/python
from dash import Dash,dcc,html,page_container,page_registry
from glob import glob
import subprocess

files = glob("./pages/*.ipynb")

for f in files:
    py_file = "." + f.split('.')[-2] + ".py"
    subprocess.run("colab-convert "+f+" "+py_file,shell=True)


app = Dash(__name__,use_pages=True,pages_folder='./pages')

app.layout = html.Div([
    html.H1("بسمله حرهما نراهم",style={'color':'green'}),
    html.Div([
        html.Div(dcc.Link(children=page['name'],href=page['relative_path'])) for page in page_registry.values()
    ]),
    page_container
])

if __name__ == '__main__':
    app.run(port=8078,debug=False)


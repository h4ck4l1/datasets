import os,sys,time
from glob import glob
from dash import Dash, html, dcc, register_page, page_container
import dash_bootstrap_components as dbc

app = Dash(
    __name__,
    assets_folder="assets",
    pages_folder="pages",
    use_pages=True,
    suppress_callback_exceptions=True
    )

def get_image(name:str,dataset_name:str=None):
    all_images = glob("assets/*.png") + \
                 glob("assets/*.jpg") + \
                 glob("assets/*.jpeg") + \
                 glob("assets/*.svg")
    
    if dataset_name:
        for img in all_images:
            img_name = dataset_name + "_" + os.path.basename(img).split(".")[0]
            if name in img_name:
                return img
    else:
        for img in all_images:
            img_name = os.path.basename(img).split(".")[0]
            if name in img_name:
                return img


dropdown_options = []
datasets_list = ["index","mushroom","covid","kfc_stock"]
for dataset in datasets_list:
    if dataset == "index":
        dropdown_options.append(
            html.Li(
                html.Span(
                    [
                        html.Img(src=get_image(dataset)),
                        dcc.Link(dataset.upper(),href="/",className="dropdown-link")
                    ]
                )
            )
        )
    else:
        dropdown_options.append(
            html.Li(
                html.Span(
                    [
                        html.Img(src=get_image(dataset)),
                        dcc.Link(dataset.upper(),href=f"/{dataset}",className="dropdown-link")
                    ]
                )
            )
        )

app.layout = dcc.Loading(
    [
        html.Div(
            [
                html.Ul(
                    html.Li(
                        html.Span(
                            [
                                html.H1("DATASET MENU",className="dropdown-button-title"),
                                html.Img(src="assets/datascience_icon.png")
                            ]
                        )
                    ),
                    className="dropdown-button"
                ),
                html.Div(html.Span(className="arrow"),className="arrow-space"),
                html.Ul(dropdown_options,className="dropdown-box")
            ],
            className="total-dropdown"
        ),
        page_container
    ],
    type="cube",
    className="loading-cube"
)

server = app.server

if __name__ == "__main__":
    app.run_server(host='0.0.0.0',port=8050,debug=True)

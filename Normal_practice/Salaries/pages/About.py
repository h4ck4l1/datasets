from dash import dcc,html,register_page


msg = '''
This is Data Science salaries dataset which is update tri-monthly and has 10 usability with no null values

Data Science Job Salaries Dataset contains 11 columns, each are:
- Work Year:          The year the salary was paid.
- Experience Level:   The experience level in the job during the year.
- Employment Type:    The type of employment for the role.
- Job Title:          The role worked in during the year.
- Salary:             The total gross salary amount paid.
- Salary Currency:    The currency of the salary paid.
- Salary in USD:      The salary in USD.
- Employee Residence: Employee's primary country of residence in during the work year.
- Remote Ratio:       The overall amount of work done remotely.
- Company Location:   The country of the employer's main office or contracting branch.
- Company Size:       The median number of people that worked for the company during the year.



__Disclaimer:__ Links to Original Notebooks [Sal.ipynb](https://github.com/h4ck4l1/datasets/blob/main/Normal_practice/Salaries/pages/sal.ipynb) [Statistical Analysis.ipynb](https://github.com/h4ck4l1/datasets/blob/main/Normal_practice/Salaries/pages/Statistical_analysis.ipynb) These are converted to .py files using colab-convert for dash to work as its only native to default python files
'''

register_page(__name__,path='/',order=1)

layout = html.Div([
    dcc.Markdown(msg,style={'font-face':'Fira Code'})
])
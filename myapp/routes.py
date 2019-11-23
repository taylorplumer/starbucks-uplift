from myapp import app
import json, plotly
from flask import render_template
from data.visualization import return_figures

@app.route('/')
@app.route('/index')
def index():

    figures = return_figures()

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index_.html',
                           ids=ids,
                           figuresJSON=figuresJSON)

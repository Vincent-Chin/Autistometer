# Bokeh basics
from bokeh.io import curdoc
from bokeh.models.widgets import Tabs
from bokeh.charts import Horizon, output_file, show

import Web.DataModel

# Each tab is drawn by one script
#from scripts.histogram import histogram_tab
#from scripts.density import density_tab
#from scripts.table import table_tab
#from scripts.draw_map import map_tab
#from scripts.routes import route_tab


data_model = Web.DataModel.DataModel()


def serve():
    global data_model

    """
    # Create each of the tabs
    tab1 = histogram_tab(flights)
    tab2 = density_tab(flights)
    tab3 = table_tab(flights)
    tab4 = map_tab(map_data, states)
    tab5 = route_tb(flights)

    # Put all the tabs into one application
    tabs = Tabs(tabs=[tab1, tab2, tab3])

    # Put the tabs in the current document for display
    curdoc().add_root(tabs)
    """

    data = dict([
        ('AAPL', AAPL['Adj Close']),
        ('Date', AAPL['Date']),
        ('MSFT', MSFT['Adj Close']),
        ('IBM', IBM['Adj Close'])]
    )

    hp = Horizon(data, x='Date', plot_width=800, plot_height=300,
                 title="horizon plot using stock inputs")

    output_file("horizon.html")

    show(hp)

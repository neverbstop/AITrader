import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import threading
from autonomous_agent import AutonomousAgent
import datetime

# This will be set by the main thread
agent_instance: AutonomousAgent = None
last_price_info = {'price': 0.0}

def create_dashboard_layout():
    return html.Div(
        className="container mx-auto p-8",
        children=[
            html.H1("AI Trading Bot Dashboard", className="text-3xl font-bold text-white mb-6"),
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # in milliseconds
                n_intervals=0
            ),
            html.Div(
                className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8",
                children=[
                    # Portfolio Value Card
                    html.Div(
                        className="card rounded-lg p-6 shadow-md",
                        children=[
                            html.Div("Portfolio Value", className="text-lg font-semibold text-gray-400"),
                            html.Div(id="portfolio-value", className="text-4xl font-bold text-green-500 mt-2"),
                        ],
                    ),
                    # Total PnL Card
                    html.Div(
                        className="card rounded-lg p-6 shadow-md",
                        children=[
                            html.Div("Total PnL", className="text-lg font-semibold text-gray-400"),
                            html.Div(id="total-pnl", className="text-4xl font-bold text-red-500 mt-2"),
                        ],
                    ),
                    # Autonomous Agent Status Card
                    html.Div(
                        className="card rounded-lg p-6 shadow-md",
                        children=[
                            html.Div("Autonomous Agent", className="text-lg font-semibold text-gray-400"),
                            html.Div(id="autonomous-status", className="text-2xl font-bold text-gray-500 mt-2"),
                            html.Button(
                                id="toggle-button",
                                className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg shadow-sm",
                            ),
                        ],
                    ),
                ],
            ),
            # Real-time Chart
            html.Div(
                className="card rounded-lg p-6 shadow-md mb-8",
                children=[
                    html.H2("Trading Performance", className="text-xl font-semibold text-gray-200 mb-4"),
                    dcc.Graph(id="performance-chart", className="w-full h-96"),
                ],
            ),
            # Trade History Table
            html.Div(
                className="card rounded-lg p-6 shadow-md",
                children=[
                    html.H2("Trade History", className="text-xl font-semibold text-gray-200 mb-4"),
                    html.Div(
                        className="overflow-x-auto",
                        children=[
                            html.Table(
                                id="trades-table",
                                className="min-w-full divide-y divide-gray-700",
                                children=[
                                    html.Thead(
                                        className="bg-gray-800",
                                        children=[
                                            html.Tr(
                                                children=[
                                                    html.Th("Date", className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider"),
                                                    html.Th("Action", className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider"),
                                                    html.Th("Price", className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider"),
                                                    html.Th("Shares", className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider"),
                                                    html.Th("PnL", className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider"),
                                                ]
                                            )
                                        ],
                                    ),
                                    html.Tbody(id="trades-table-body", className="bg-gray-900 divide-y divide-gray-700"),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )

def register_callbacks(app: dash.Dash):
    @app.callback(
        [
            Output('portfolio-value', 'children'),
            Output('total-pnl', 'children'),
            Output('total-pnl', 'className'),
            Output('autonomous-status', 'children'),
            Output('autonomous-status', 'className'),
            Output('toggle-button', 'children'),
            Output('toggle-button', 'className'),
            Output('performance-chart', 'figure'),
            Output('trades-table-body', 'children'),
        ],
        [Input('interval-component', 'n_intervals')]
    )
    def update_metrics(n):
        if agent_instance is None:
            return dash.no_update

        current_price = last_price_info.get('price', 0.0)
        portfolio_value = agent_instance.get_portfolio_value(current_price)
        total_pnl = portfolio_value - agent_instance.budget

        pnl_class = "text-4xl font-bold mt-2 "
        if total_pnl > 0:
            pnl_class += "text-green-500"
        elif total_pnl < 0:
            pnl_class += "text-red-500"
        else:
            pnl_class += "text-gray-500"

        if agent_instance.enabled:
            status_text, status_class, button_text, button_class = (
                "ENABLED", "text-2xl font-bold text-green-500 mt-2", "Disable",
                "mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-semibold rounded-lg shadow-sm"
            )
        else:
            status_text, status_class, button_text, button_class = (
                "DISABLED", "text-2xl font-bold text-gray-500 mt-2", "Enable",
                "mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg shadow-sm"
            )

        table_rows = []
        for trade in agent_instance.trades:
            pnl_val = trade.get('pnl')
            pnl_str = f"${pnl_val:.2f}" if pnl_val is not None and pnl_val > 0 else (f"-$" + f"{-pnl_val:.2f}" if pnl_val is not None and pnl_val < 0 else "-")
            pnl_color_class = "text-gray-500"
            if pnl_val is not None:
                if pnl_val > 0: pnl_color_class = "text-green-500"
                elif pnl_val < 0: pnl_color_class = "text-red-500"

            table_rows.append(html.Tr([
                html.Td(trade['date'].strftime('%Y-%m-%d %H:%M:%S'), className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-300"),
                html.Td(html.Span(trade['action'], className=f"font-semibold {'text-green-500' if 'BUY' in trade['action'] else 'text-red-500'}"), className="px-6 py-4 whitespace-nowrap text-sm text-gray-300"),
                html.Td(f"${trade['price']:.2f}", className="px-6 py-4 whitespace-nowrap text-sm text-gray-300"),
                html.Td(trade['shares'], className="px-6 py-4 whitespace-nowrap text-sm text-gray-300"),
                html.Td(pnl_str, className=f"px-6 py-4 whitespace-nowrap text-sm font-medium {pnl_color_class}"),
            ]))

        portfolio_history = [{'date': datetime.datetime.now() - datetime.timedelta(minutes=1), 'value': agent_instance.budget}]
        cumulative_pnl = 0
        for trade in agent_instance.trades:
            if trade.get('pnl') is not None:
                cumulative_pnl += trade['pnl']
            portfolio_history.append({'date': trade['date'], 'value': agent_instance.budget + cumulative_pnl})
        
        if agent_instance.is_in_position:
             portfolio_history.append({'date': datetime.datetime.now(), 'value': portfolio_value})

        portfolio_df = pd.DataFrame(portfolio_history)

        chart_data = [go.Scatter(x=portfolio_df['date'], y=portfolio_df['value'], mode='lines+markers', name='Portfolio Value', line={'color': '#10b981'})]
        
        layout = {
            'paper_bgcolor': '#161b22', 'plot_bgcolor': '#161b22',
            'font': {'color': '#c9d1d9'}, 'margin': {'t': 20, 'b': 40, 'l': 60, 'r': 20},
            'xaxis': {'title': 'Date', 'gridcolor': '#30363d'}, 'yaxis': {'title': 'Value ($)', 'gridcolor': '#30363d'}
        }
        figure = {'data': chart_data, 'layout': layout}

        return (
            f"${portfolio_value:.2f}", f"${total_pnl:.2f}", pnl_class,
            status_text, status_class, button_text, button_class,
            figure, table_rows
        )

    @app.callback(
        Output('toggle-button', 'n_clicks'),
        [Input('toggle-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_agent_status(n_clicks):
        if agent_instance is not None:
            if agent_instance.enabled:
                agent_instance.disable()
            else:
                agent_instance.enable()
        return dash.no_update

def run_dashboard(agent: AutonomousAgent, price_info: dict):
    """
    Runs the trading dashboard in a separate thread.
    """
    global agent_instance, last_price_info
    agent_instance = agent
    last_price_info = price_info

    app = dash.Dash(__name__, external_scripts=['https://cdn.tailwindcss.com'])
    app.title = "AI Trading Bot Dashboard"
    
    # Custom HTML to include dark mode and font
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    font-family: 'Inter', sans-serif;
                    background-color: #0d1117;
                    color: #c9d1d9;
                }
                .card {
                    background-color: #161b22;
                    border: 1px solid #30363d;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    app.layout = create_dashboard_layout()
    register_callbacks(app)

    def run_app():
        # In a real application, you would use a production-ready WSGI server
        # like gunicorn or waitress instead of the development server.
        app.run_server(debug=False, host='0.0.0.0', port=8050)

    dashboard_thread = threading.Thread(target=run_app)
    dashboard_thread.daemon = True
    dashboard_thread.start()
    print("📈 Dashboard is running on http://127.0.0.1:8050")

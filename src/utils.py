import plotly.express as px
import plotly.graph_objects as go

def bar_plot(
            x_data: list,
            y_data: list,
            title: str,
            x_title: str,
            y_title: str,
            output_path: str,
            ):
    trace = go.Bar(x = x_data,y = y_data)
    data=[trace]
    # defining layout
    layout = go.Layout(title=title,
                        xaxis=dict(title=x_title),
                        yaxis=dict(title=y_title),
                        hovermode='closest')
    # defining figure and plotting
    figure = go.Figure(data = data,layout = layout)
    figure.write_html(output_path, auto_open = False)

def scatter_plot(
            x_data: list,
            y_data: list,
            title: str,
            x_title: str,
            y_title: str,
            output_path: str,
            color_text: str = None,
            ):
    if color_text is None:
        trace = go.Scatter(x=x_data,y=y_data)
    else:
        trace = go.Scatter(x=x_data,
                           y=y_data,
                           marker=dict(color=color_text,
                                        colorscale='viridis',
                                        showscale = True),
                           text=color_text,
                           mode='lines+markers')    
    data=[trace]
    # defining layout
    layout = go.Layout(title=title,
                        xaxis=dict(title=x_title),
                        yaxis=dict(title=y_title),
                        hovermode='closest')
    # defining figure and plotting
    figure = go.Figure(data=data,layout=layout)
    figure.write_html(output_path, auto_open = False)

if __name__ == '__main__':
    x_data = [345,56,67,123,676,234,67,789,345]
    y_data = [1,2,3,5,6,7,8,9]
    color_text = [1,1,1,0,0,0,1,0,1]
    title = "test"
    x_title = "x"
    y_title = "y"
    output_path = "test.html"
    scatter_plot(x_data=x_data,
             y_data=y_data,
             title=title,
             x_title=x_title,
             y_title=y_title,
             output_path=output_path,
             color_text=color_text)
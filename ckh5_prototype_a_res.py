import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv(url, names=names)

# Preprocess data
X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create dashboard
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Machine Learning Model Dashboard'),
    html.Hr(),
    html.Div([
        html.P('Select a feature to visualize:'),
        dcc.Dropdown(
            id='feature-dropdown',
            options=[{'label': i, 'value': i} for i in X.columns],
            value=X.columns[0]
        )
    ]),
    dcc.Graph(id='feature-graph'),
    html.Hr(),
    html.H2('Model Performance'),
    html.Div([
        html.P('Accuracy:'),
        html.P(id='accuracy')
    ]),
    html.Div([
        html.P('Classification Report:'),
        html.Pre(id='classification-report')
    ]),
    html.Div([
        html.P('Confusion Matrix:'),
        html.Pre(id='confusion-matrix')
    ])
])

@app.callback(
    Output('feature-graph', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_graph(feature):
    fig = px.histogram(data, x=feature, color='class', barmode='group')
    return fig

@app.callback(
    [Output('accuracy', 'children'),
     Output('classification-report', 'children'),
     Output('confusion-matrix', 'children')],
    [Input('feature-dropdown', 'value')]
)
def update_performance(feature):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)
    return f'Accuracy: {accuracy:.3f}', f'{report}', f'{matrix}'

if __name__ == '__main__':
    app.run_server(debug=True)
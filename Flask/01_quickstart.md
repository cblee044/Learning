# Quickstart

## A Minimal Application
Saving this as `hello.py` is enough to run a flask application

```python
from flask import Flask

# Create an instance of Flask. The first argument is the name of the application's module.
# It also tells Flask where to look for templates and static files
app = Flask(__name__)

# The route() decorator sets up a URL to trigger the function. 
@app.route('/')
def index():
  return "<H1>This is Flask</H1>"
```

### Run the application
Before running the application, the `FLASK_APP` environment variable must be set to the name of the application.

```
$ export FLASK_APP=hello
$ flask run
```

> Application Discovery Behavior: As a shortcut, if the file is named app.py or wsgi.py, you don’t have to set the FLASK_APP environment variable. See [Command Line Interface](https://flask.palletsprojects.com/en/2.1.x/cli/) for more details.

### Debug Mode
When activated, the debugger will automatically reload the server on saved changes and present itself interactively if an error comes up. Enter `development` mode to enable the debugger.

```
$ export FLASK_ENV=development
$ flask run
```

### Variable Rules

Arguments are passed from the website URL into its corresponding function using `<variable_name>`

```python
from markupsafe import escape

@app.route('/user/<variable_name>')
def homepage(variable_name):
  return(f'Look!, the variable from the URL is {variable_name}')
```

### Unique URLs / Redirection Behavior
If a function is routed to `'/projects/'` then a user will trigger that function with or without including that last backlash. However, a route to `'/projects'` will trigger a 404 "Not Found" error if a backslash is included while accessing the URL.
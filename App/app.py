from flask import Flask
from routes.main_routes import main_blueprint
from routes.auth_routes import auth_blueprint  
from routes.crawlfetch_routes import crawlfetch_blueprint  
from routes.admin_routes import admin_blueprint
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)
app.config.from_pyfile('config.py')

# Register Blueprints
app.register_blueprint(main_blueprint)
app.register_blueprint(auth_blueprint)  
app.register_blueprint(crawlfetch_blueprint) 
app.register_blueprint(admin_blueprint)

if __name__ == '__main__':
    app.run(debug=True)

# Imports
from flask import Flask
from . import pages        # Import the blueprint from pages.py
import os


def create_app():
    """
    Application factory function
    Creates and configures a Flask app instance,
    then register all blueprints (routes)
    """

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # Go up a level
    app = Flask(
        __name__,
        template_folder=os.path.join(project_root, "board", "templates"),
        static_folder=os.path.join(project_root, "uploads")
    )
    
    # Register the 'pages' blueprint
    app.register_blueprint(pages.bp)
    return app


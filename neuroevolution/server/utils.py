# server/utils.py
# Add any utility functions here
from neuroevolution.server.server import app
from flask import jsonify
from neuroevolution.server.errors import InvalidMediatorIdError, MissingFieldsError

@app.errorhandler(InvalidMediatorIdError)
def handle_invalid_mediator_id(error):
    return jsonify({'error': str(error)}), 422

@app.errorhandler(MissingFieldsError)
def handle_missing_fields(error):
    return jsonify({'error': str(error)}), 422
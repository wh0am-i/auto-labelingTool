from flask import Flask, jsonify, request
from functools import wraps
from label_studio_ml.examples.segment_anything_2_image.model import NewModel
import os

app = Flask(__name__)

# Instância do modelo
model = NewModel()

# Load expected token from environment
EXPECTED_TOKEN = os.getenv('API_KEY') or os.getenv('LABEL_STUDIO_API_KEY') or os.getenv('API_PERSONAL_ACESS_TOKEN')

# Função para verificar o token
def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        auth = request.headers.get('Authorization')
        if not auth:
            return jsonify({"error": "Token de autenticação não fornecido!"}), 401

        # Accept "Bearer <token>" or "Token <token>" or raw token
        token = auth
        if auth.startswith('Bearer '):
            token = auth.split(' ', 1)[1]
        elif auth.startswith('Token '):
            token = auth.split(' ', 1)[1]

        if not EXPECTED_TOKEN:
            return jsonify({"error": "Server token not configured"}), 500

        if token != EXPECTED_TOKEN:
            return jsonify({"error": "Token inválido!"}), 401

        return f(*args, **kwargs)

    return decorator

@app.route('/projects', methods=['GET'])
@token_required
def list_projects():
    """Endpoint para listar todos os projetos disponíveis"""
    try:
        projects = model.list_projects()
        return jsonify({"projects": projects}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/auto-label', methods=['POST'])
@token_required
def auto_label_project():
    """Endpoint para realizar auto-labeling em um projeto selecionado"""
    try:
        project_id = request.json.get('project_id')
        if not project_id:
            return jsonify({"error": "O ID do projeto é necessário"}), 400
        
        # Iniciar auto-labeling para o projeto
        model.auto_label(project_id)
        return jsonify({"message": f"Auto-labeling iniciado para o projeto {project_id}."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar se o servidor está rodando"""
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9090, debug=True)

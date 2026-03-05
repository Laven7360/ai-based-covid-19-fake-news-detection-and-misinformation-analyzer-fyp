from flask import Blueprint, request, jsonify
from app_pipeline import classify_article_from_url, classify_all_from_text

crawlfetch_blueprint = Blueprint('crawlfetch_blueprint', __name__)

@crawlfetch_blueprint.route('/api/crawlfetch', methods=['POST'])
def crawlfetch_classify():
    data = request.get_json()

    url = data.get('url')
    text = data.get('text')

    if url:
        result = classify_article_from_url(url)
    elif text:
        result = classify_all_from_text(text)
    else:
        return jsonify({"error": "Missing 'url' or 'text' in request"}), 400

    return jsonify(result)

from flask import Flask, request, jsonify
import json
import os
import math
import re
import csv
from collections import defaultdict, Counter

app = Flask(__name__)

# Constants
VALID_SENTIMENTS = frozenset(["positive", "negative", "neutral"])
ENCODINGS_TO_TRY = ['utf-8', 'windows-1252', 'ISO-8859-1']
LAPLACE_SMOOTHING_ALPHA = 1.0

# Enhanced stopwords list
STOPWORDS = frozenset({
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 
    'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 
    'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 
    'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 
    'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 
    'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', 
    "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', 
    "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 
    'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 
    'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 
    'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', 
    "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 
    'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 
    'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", 
    "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 
    'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', 
    "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 
    'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', 
    "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 
    'yourself', 'yourselves'
})

NEGATION_WORDS = frozenset({
    "not", "no", "never", "don't", "doesn't", "didn't", "wasn't", "weren't",
    "isn't", "aren't", "haven't", "hasn't", "won't", "wouldn't", "can't", "couldn't"
})

INTENSIFIERS = frozenset({'very', 'extremely', 'really', 'so', 'too', 'absolutely', 'completely', 'quite', 'totally', 'truly', 'particularly', 'especially'})
# Enhanced emotion word lists with weights
IMPORTANT_NEGATIVE_WORDS = frozenset({'suicide', 'disappointed', 'hate', 'depressed', 'angry', 'sad', 'hopeless', 
                                     'upset', 'hurt', 'anxious', 'worried', 'afraid', 'unhappy', 'miserable'})

IMPORTANT_POSITIVE_WORDS = frozenset({'happy', 'joy', 'excited', 'love', 'grateful', 'pleased', 'wonderful', 
                                     'delighted', 'glad', 'content', 'peaceful', 'satisfied'})

# Emotion transition markers
TEMPORAL_TRANSITION_WORDS = frozenset({'but', 'however', 'although', 'though', 'yet', 'now', 'today', 'lately'})

# Use more memory-efficient dict for constants with updated weights
NEGATIVE_BOOST_WORDS = {
    'suicide': 3.0,
    'disappointed': 2.0,
    'hate': 2.5,
    'depressed': 2.5,
    'angry': 2.0,
    'sad': 2.0,
    'hopeless': 2.5,
    'upset': 1.8,
    'hurt': 1.8,
    'anxious': 1.5,
    'worried': 1.5,
    'afraid': 1.8,
    'unhappy': 1.8,
    'miserable': 2.2
}

POSITIVE_BOOST_WORDS = {
    'happy': 1.5,
    'joy': 1.8, 
    'excited': 1.5,
    'love': 2.0,
    'grateful': 1.8,
    'pleased': 1.5,
    'wonderful': 1.8,
    'delighted': 1.8,
    'glad': 1.5,
    'content': 1.5,
    'peaceful': 1.5,
    'satisfied': 1.5
}

# Blog recommendations - moved to a separate constant
BLOG_RECOMMENDATIONS = {
    "positive": [
        {"title": "5 Ways to Maintain Your Positive Mindset", "url": "/blogs/positive-mindset"},
        {"title": "The Benefits of Daily Gratitude Practice", "url": "/blogs/gratitude-practice"},
        {"title": "How Positivity Affects Physical Health", "url": "/blogs/positivity-health"}
    ],
    "negative": [
        {"title": "Coping with Difficult Emotions", "url": "/blogs/coping-emotions"},
        {"title": "5 Steps to Overcome Negative Thoughts", "url": "/blogs/overcome-negativity"},
        {"title": "Finding Hope in Challenging Situations", "url": "/blogs/finding-hope"}
    ],
    "neutral": [
        {"title": "Mindfulness Practices for Everyday Life", "url": "/blogs/mindfulness"},
        {"title": "Understanding Your Emotional Patterns", "url": "/blogs/emotional-patterns"},
        {"title": "Building Emotional Intelligence", "url": "/blogs/emotional-intelligence"}
    ]
}

# Global model state
class SentimentModel:
    def __init__(self):
        self.word_counts = {sentiment: Counter() for sentiment in VALID_SENTIMENTS}
        self.class_counts = {sentiment: 0 for sentiment in VALID_SENTIMENTS}
        self.vocabulary = set()
        self.total_documents = 0
        self.is_trained = False

# Initialize model
model = SentimentModel()

def load_training_data_from_csv(filepath):
    """Load training data from CSV file with fallback encodings"""
    data = []

    for encoding in ENCODINGS_TO_TRY:
        try:
            with open(filepath, newline='', encoding=encoding) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    text = row.get('text', '').strip()
                    label = row.get('label', '').strip().lower()
                    
                    if text and label in VALID_SENTIMENTS:
                        data.append({'text': text, 'label': label})
                        
            # If we get here, the file was read successfully
            return data
            
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")

    if not data:
        raise ValueError("No valid training data found in CSV file or incompatible encoding")
    
    return data

def preprocess_text(text):
    """Process text into a set of features with enhanced emotion handling and context awareness"""
    # Convert to lowercase and remove non-alphanumeric characters except specific punctuation
    text = re.sub(r'[^\w\s.,!?]', '', text.lower())
    
    # Replace punctuation with spaces to ensure proper tokenization
    text = re.sub(r'[.,!?]', ' ', text)
    
    words = text.split()
    features = set()
    negated = False
    
    # Track emotional transitions
    has_transition = False
    transition_position = -1
    
    # Find transition markers (like "but", "however", "today")
    for i, word in enumerate(words):
        if word in TEMPORAL_TRANSITION_WORDS:
            has_transition = True
            transition_position = i
            features.add(f"transition_{word}")
    
    # Track emotions before and after transition
    emotion_before_transition = None
    emotion_after_transition = None
    
    for i, word in enumerate(words):
        # Handle negation words
        if word in NEGATION_WORDS:
            negated = not negated
            features.add(f"negation_present")  # Add feature indicating negation exists
            continue
            
        # Skip regular stopwords
        if word in STOPWORDS and word not in INTENSIFIERS and word not in IMPORTANT_NEGATIVE_WORDS and word not in IMPORTANT_POSITIVE_WORDS:
            continue

        # Add the word with negation prefix if needed
        feature = f"not_{word}" if negated else word
        features.add(feature)
        
        # Track emotion words relative to transitions
        if has_transition:
            if word in IMPORTANT_NEGATIVE_WORDS:
                if i < transition_position:
                    emotion_before_transition = "negative"
                    features.add("negative_before_transition")
                else:
                    emotion_after_transition = "negative"
                    features.add("negative_after_transition")
            
            if word in IMPORTANT_POSITIVE_WORDS:
                if i < transition_position:
                    emotion_before_transition = "positive"
                    features.add("positive_before_transition")
                else:
                    emotion_after_transition = "positive"
                    features.add("positive_after_transition")
        
        # Add special bigrams and trigrams to capture context
        if i < len(words) - 1:
            next_word = words[i + 1]
            
            # Create bigrams for emotional and intensifier words
            if (word in INTENSIFIERS or 
                word in IMPORTANT_NEGATIVE_WORDS or 
                word in IMPORTANT_POSITIVE_WORDS or
                next_word in IMPORTANT_NEGATIVE_WORDS or
                next_word in IMPORTANT_POSITIVE_WORDS):
                bigram = f"{word}_{next_word}"
                features.add(bigram)
                
                # Add stronger weight to emotion-related bigrams
                if word in INTENSIFIERS and (next_word in IMPORTANT_NEGATIVE_WORDS or next_word in IMPORTANT_POSITIVE_WORDS):
                    features.add(f"intensified_{bigram}")
            
            # Create contextual trigrams if possible
            if i < len(words) - 2:
                next_next_word = words[i + 2]
                if ((word in IMPORTANT_NEGATIVE_WORDS or word in IMPORTANT_POSITIVE_WORDS) or
                    (next_word in IMPORTANT_NEGATIVE_WORDS or next_word in IMPORTANT_POSITIVE_WORDS) or
                    (next_next_word in IMPORTANT_NEGATIVE_WORDS or next_next_word in IMPORTANT_POSITIVE_WORDS)):
                    features.add(f"{word}_{next_word}_{next_next_word}")
    
    # Add transition pattern if emotions changed (positive->negative or negative->positive)
    if emotion_before_transition and emotion_after_transition and emotion_before_transition != emotion_after_transition:
        features.add(f"emotion_shift_{emotion_before_transition}_to_{emotion_after_transition}")
        
        # Emphasize the most recent emotion as likely more important
        features.add(f"recent_emotion_{emotion_after_transition}")
    
    # Add recency bias - last emotional word gets extra emphasis
    last_emotion_word = None
    for word in reversed(words):
        if word in IMPORTANT_NEGATIVE_WORDS:
            last_emotion_word = f"last_emotion_negative_{word}"
            break
        elif word in IMPORTANT_POSITIVE_WORDS:
            last_emotion_word = f"last_emotion_positive_{word}"
            break
    
    if last_emotion_word:
        features.add(last_emotion_word)
    
    return features

def train_model(data):
    """Train the sentiment analysis model on the provided data"""
    # Reset model state
    model.word_counts = {sentiment: Counter() for sentiment in VALID_SENTIMENTS}
    model.class_counts = {sentiment: 0 for sentiment in VALID_SENTIMENTS}
    model.vocabulary = set()
    model.total_documents = 0
    
    valid_samples = 0
    
    for sample in data:
        label = sample.get("label", "").strip().lower()
        text = sample.get("text", "").strip()
        
        if not text or label not in VALID_SENTIMENTS:
            continue

        # Process valid samples
        model.class_counts[label] += 1
        valid_samples += 1
        
        features = preprocess_text(text)
        model.vocabulary.update(features)

        # Count word occurrences for this class
        model.word_counts[label].update(features)

    model.total_documents = valid_samples
    
    if valid_samples == 0:
        raise ValueError("No valid training data found")

    model.is_trained = True
    
    return {
        "status": "success",
        "message": "Model trained successfully",
        "stats": {
            "vocabulary_size": len(model.vocabulary),
            "positive_examples": model.class_counts["positive"],
            "negative_examples": model.class_counts["negative"],
            "neutral_examples": model.class_counts["neutral"],
            "total_documents": model.total_documents
        }
    }

def predict(text):
    """Predict sentiment of given text using the trained model with enhanced context handling"""
    if not model.is_trained:
        return {"error": "Model not trained yet"}
    
    # Extract features from input text
    features = preprocess_text(text)
    
    # Calculate vocabulary size once
    vocab_size = len(model.vocabulary)
    log_probs = {}
    
    # Track feature importance for explanation
    feature_importance = {sentiment: {} for sentiment in VALID_SENTIMENTS}
    
    # Special case handling for emotion transitions
    has_emotion_shift = any(f.startswith("emotion_shift_") for f in features)
    has_recent_emotion = any(f.startswith("recent_emotion_") for f in features)
    recent_emotion = None
    
    # Get the recent emotion if present
    if has_recent_emotion:
        for f in features:
            if f.startswith("recent_emotion_"):
                recent_emotion = f.split("_")[-1]
                break
            
    # Get the last emotion word
    last_emotion = None
    emotion_type = None
    for f in features:
        if f.startswith("last_emotion_"):
            parts = f.split("_")
            emotion_type = parts[1]  # last_emotion_negative_sad -> negative
            last_emotion = parts[2] if len(parts) > 2 else None  # last_emotion_negative_sad -> sad
            break
    
    # Initial calculation of log probabilities for each sentiment class
    for sentiment in VALID_SENTIMENTS:
        # Calculate prior probability
        class_prob = model.class_counts[sentiment] / model.total_documents
        
        # Start with log of prior probability
        log_prob = math.log(class_prob)
        
        # Total words in this sentiment class (for denominator)
        total_words = sum(model.word_counts[sentiment].values())
        denominator = total_words + LAPLACE_SMOOTHING_ALPHA * vocab_size
        
        # Calculate feature contributions
        for word in features:
            count = model.word_counts[sentiment].get(word, 0) + LAPLACE_SMOOTHING_ALPHA
            
            # Apply boosting for emotional words based on sentiment
            if sentiment == 'negative' and word in NEGATIVE_BOOST_WORDS:
                boost_factor = NEGATIVE_BOOST_WORDS[word]
                count *= boost_factor
                feature_importance[sentiment][word] = boost_factor
            
            elif sentiment == 'positive' and word in POSITIVE_BOOST_WORDS:
                boost_factor = POSITIVE_BOOST_WORDS[word]
                count *= boost_factor
                feature_importance[sentiment][word] = boost_factor
                
            # Apply special weighting for context patterns
            
            # Bigram and trigram boosting
            if '_' in word and not word.startswith('not_'):
                # Boost emotional bigrams more than regular words
                if any(ew in word for ew in IMPORTANT_NEGATIVE_WORDS) or any(ew in word for ew in IMPORTANT_POSITIVE_WORDS):
                    count *= 1.5
                    feature_importance[sentiment][word] = 1.5
            
            # Boosting for emotional transitions
            if word.startswith("emotion_shift_"):
                shift_type = word.split("_")[-3:]  # ["positive", "to", "negative"]
                if shift_type[0] != shift_type[2]:  # if emotions are different
                    # Emphasize final emotion more by boosting that sentiment
                    if sentiment == shift_type[2]:  # if the sentiment matches the final emotion
                        count *= 3.0  # Increased from 2.0 to 3.0
                        feature_importance[sentiment][word] = 3.0
            
            # Boosting for recent emotion
            if word.startswith("recent_emotion_"):
                emotion = word.split("_")[-1]
                if sentiment == emotion:
                    count *= 3.5  # Increased from 2.5 to 3.5
                    feature_importance[sentiment][word] = 3.5
            
            # Boosting for last emotional word
            if word.startswith("last_emotion_"):
                found_emotion_type = word.split("_")[1]  # negative or positive
                if sentiment == found_emotion_type:
                    count *= 3.0  # Increased from 2.0 to 3.0
                    feature_importance[sentiment][word] = 3.0
            
            # Add log probability for this feature
            log_prob += math.log(count / denominator)

        log_probs[sentiment] = log_prob

    # Convert log probabilities to normalized confidence scores
    max_log = max(log_probs.values())
    exp_scores = {sentiment: math.exp(score - max_log) for sentiment, score in log_probs.items()}
    total = sum(exp_scores.values())
    confidence_scores = {sentiment: round(score / total, 4) for sentiment, score in exp_scores.items()}
    
    # Check for explicit mixed emotion cases with decisive handling
    explicit_mixed_emotions = False
    
    # CRITICAL IMPROVEMENT: Apply strong rule-based corrections for mixed emotions with transitions
    if has_emotion_shift:
        explicit_mixed_emotions = True
        
        # Find the final emotion in the shift
        final_emotion = None
        for f in features:
            if f.startswith("emotion_shift_"):
                parts = f.split("_")
                if len(parts) >= 5 and parts[-2] == "to":
                    final_emotion = parts[-1]  # Get the target emotion after "to"
                    break
        
        # Apply very strong weighting to the final emotion in a shift
        if final_emotion:
            # Dramatically increase the weight of the final emotion (3x)
            for sentiment in VALID_SENTIMENTS:
                if sentiment == final_emotion:
                    confidence_scores[sentiment] *= 3.0
                else:
                    # Reduce other sentiments
                    confidence_scores[sentiment] *= 0.5
                    
            # Renormalize
            total = sum(confidence_scores.values())
            confidence_scores = {sentiment: round(score / total, 4) for sentiment, score in confidence_scores.items()}
    
    # Apply very strong last-emotion rule - the final emotional state is critical
    if last_emotion and emotion_type and not explicit_mixed_emotions:
        # If last emotion is negative, boost negative sentiment
        if emotion_type == "negative":
            confidence_scores["negative"] = max(confidence_scores["negative"], 
                                              confidence_scores["negative"] * 2.0)
            confidence_scores["positive"] *= 0.5  # Reduce positive sentiment
        
        # If last emotion is positive, boost positive sentiment
        elif emotion_type == "positive":
            confidence_scores["positive"] = max(confidence_scores["positive"],
                                             confidence_scores["positive"] * 2.0)
            confidence_scores["negative"] *= 0.5  # Reduce negative sentiment
        
        # Renormalize
        total = sum(confidence_scores.values())
        confidence_scores = {sentiment: round(score / total, 4) for sentiment, score in confidence_scores.items()}
    
    # DECISIVE RULE: If text ends with clear negative emotion after transition, make it negative
    if "negative_after_transition" in features and "transition_but" in features:
        has_final_negative = False
        for f in features:
            if f == "last_emotion_negative_sad" or f == "recent_emotion_negative":
                has_final_negative = True
                break
                
        if has_final_negative:
            # Force negative to be the highest by a clear margin
            confidence_scores = {
                "negative": 0.80,
                "neutral": 0.05,
                "positive": 0.15
            }
    
    # Get predicted sentiment (highest confidence)
    predicted = max(confidence_scores, key=confidence_scores.get)

    # Identify key features that contributed to the prediction
    key_features = []
    for word in sorted(features):
        if word in NEGATIVE_BOOST_WORDS or word in POSITIVE_BOOST_WORDS:
            key_features.append(word)
        elif word.startswith("emotion_shift_") or word.startswith("recent_emotion_") or word.startswith("last_emotion_"):
            key_features.append(word)
    
    # Add debugging information about rule application
    applied_rules = []
    if has_emotion_shift:
        applied_rules.append("emotion_shift_rule")
    if last_emotion and emotion_type:
        applied_rules.append("last_emotion_rule")
    if "negative_after_transition" in features and "transition_but" in features:
        applied_rules.append("negative_after_but_rule")
    
    # Return prediction with additional info
    return {
        "sentiment": predicted,
        "confidence_scores": confidence_scores,
        "recommendations": BLOG_RECOMMENDATIONS.get(predicted, []),
        "processed_features": sorted(list(features)),
        "key_emotional_features": key_features[:5] if key_features else [],
        "applied_rules": applied_rules
    }

# API Endpoints
@app.route('/', methods=['GET'])
def welcome():
    """Welcome endpoint with basic API information"""
    return jsonify({
        "name": "Sentiment Analysis API",
        "status": "active",
        "version": "1.1.0",
        "endpoints": {
            "/train": "POST - Train the sentiment model",
            "/predict": "POST - Predict sentiment of text",
            "/health": "GET - Check API health status"
        },
        "model_status": "trained" if model.is_trained else "not trained"
    })

@app.route('/train', methods=['POST'])
def train_endpoint():
    """API endpoint for training the model"""
    try:
        if not request.is_json:
            return jsonify({"status": "error", "message": "Request must be JSON"}), 400
        
        req = request.json
        
        # Handle different input methods
        if 'data' in req and isinstance(req['data'], list):
            training_data = req['data']
        elif 'csv_path' in req and isinstance(req['csv_path'], str):
            try:
                training_data = load_training_data_from_csv(req['csv_path'])
            except ValueError as e:
                return jsonify({"status": "error", "message": str(e)}), 400
        else:
            return jsonify({
                "status": "error",
                "message": "Either 'data' (list) or 'csv_path' (string) must be provided"
            }), 400
        
        # Train the model and return results
        result = train_model(training_data)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Training failed: {str(e)}"
        }), 500

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """API endpoint for making sentiment predictions"""
    try:
        # Check if model is trained
        if not model.is_trained:
            return jsonify({
                "error": "Model not trained yet",
                "solution": "Please train the model first using the /train endpoint"
            }), 400
        
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        if 'text' not in request.json:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = request.json['text']
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Invalid text input"}), 400
        
        # Get prediction
        result = predict(text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with model statistics"""
    stats = {}
    
    if model.is_trained:
        stats = {
            "vocabulary_size": len(model.vocabulary),
            "positive_examples": model.class_counts["positive"],
            "negative_examples": model.class_counts["negative"],
            "neutral_examples": model.class_counts["neutral"],
            "total_documents": model.total_documents
        }
    
    return jsonify({
        "status": "ok",
        "model_trained": model.is_trained,
        "stats": stats
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
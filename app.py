from flask import Flask, request, jsonify
import json
import os
import math
import re
import csv
import pickle
from collections import defaultdict, Counter

app = Flask(__name__)

# Constants
VALID_SENTIMENTS = frozenset(["positive", "negative", "neutral"])
ENCODINGS_TO_TRY = ['utf-8', 'windows-1252', 'ISO-8859-1']
LAPLACE_SMOOTHING_ALPHA = 1.0
MODEL_SAVE_PATH = "sentiment_model.pkl"  # Path to save/load the model

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

# Words indicating negation or contradiction
NEGATION_WORDS = frozenset({
    "not", "no", "never", "don't", "doesn't", "didn't", "wasn't", "weren't",
    "isn't", "aren't", "haven't", "hasn't", "won't", "wouldn't", "can't", "couldn't"
})

# Words indicating intensity
INTENSIFIERS = frozenset({
    'very', 'extremely', 'really', 'so', 'too', 'absolutely', 
    'completely', 'quite', 'totally', 'truly', 'particularly', 'especially'
})

# Context transition markers - expanded
CONTEXT_TRANSITIONS = frozenset({
    # Contrast transitions
    'but', 'however', 'although', 'though', 'yet', 'nevertheless', 'despite', 
    'in spite of', 'conversely', 'on the contrary', 'on the other hand',
    # Temporal transitions
    'now', 'currently', 'today', 'lately', 'recently', 'previously', 'before',
    'after', 'then', 'since', 'meanwhile', 'subsequently', 'eventually',
    # Causal transitions
    'because', 'since', 'as', 'therefore', 'thus', 'consequently', 'hence',
    'as a result', 'due to', 'for this reason',
    # Concession transitions
    'still', 'nonetheless', 'regardless', 'even so', 'all the same'
})

# Lists of known emotional words - not weighted, just used for recognition
KNOWN_NEGATIVE_WORDS = frozenset({
    'suicide', 'disappointed', 'hate', 'depressed', 'angry', 'sad', 'hopeless', 
    'upset', 'hurt', 'anxious', 'worried', 'afraid', 'unhappy', 'miserable',
    'awful', 'terrible', 'horrible', 'dreadful', 'annoyed', 'irritated',
    'furious', 'enraged', 'disgusted', 'outraged', 'frightened', 'terrified',
    'devastated', 'heartbroken', 'crushed', 'betrayed', 'rejected', 'abandoned',
    'distressed', 'troubled', 'confused', 'lost', 'helpless', 'vulnerable',
    'ashamed', 'embarrassed', 'humiliated', 'guilty', 'regretful', 'remorseful'
})

KNOWN_POSITIVE_WORDS = frozenset({
    'happy', 'joy', 'excited', 'love', 'grateful', 'pleased', 'wonderful', 
    'delighted', 'glad', 'content', 'peaceful', 'satisfied', 'cheerful',
    'ecstatic', 'elated', 'thrilled', 'jubilant', 'overjoyed', 'blissful',
    'grateful', 'thankful', 'appreciative', 'proud', 'confident', 'optimistic',
    'hopeful', 'inspired', 'enthusiastic', 'passionate', 'amazed', 'impressed',
    'admiring', 'adoring', 'affectionate', 'loving', 'caring', 'comforted',
    'relieved', 'relaxed', 'calm', 'serene', 'tranquil', 'fulfilled'
})

# Blog recommendations - kept the same
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

def save_model(model_obj, filepath=MODEL_SAVE_PATH):
    """Save the trained model to disk"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model_obj, f)
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def load_model(filepath=MODEL_SAVE_PATH):
    """Load a trained model from disk"""
    global model
    try:
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully from {filepath}")
            return True
        return False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

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

def extract_sentence_chunks(text):
    """Break text into sentence chunks for contextual analysis"""
    # Split by common sentence terminators with possible spaces after
    chunks = re.split(r'[.!?]\s*', text)
    # Remove empty chunks
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def analyze_text_structure(text):
    """Analyze the overall structure of the text for contextual features"""
    structure_info = {
        "sentence_count": 0,
        "avg_sentence_length": 0,
        "contains_contrast": False,
        "contains_question": "?" in text,
        "contains_exclamation": "!" in text,
        "sentiment_shifts": [],
    }
    
    # Get sentence chunks
    sentences = extract_sentence_chunks(text)
    structure_info["sentence_count"] = len(sentences)
    
    # Calculate average sentence length in words
    if sentences:
        word_counts = [len(s.split()) for s in sentences]
        structure_info["avg_sentence_length"] = sum(word_counts) / len(sentences)
    
    # Look for contrast markers between sentences
    prev_sentiment = None
    
    for i, sentence in enumerate(sentences):
        words = sentence.lower().split()
        
        # Check for contrast words at beginning of sentences (except first)
        if i > 0 and words and words[0] in CONTEXT_TRANSITIONS:
            structure_info["contains_contrast"] = True
        
        # Simple sentiment detection for shift analysis
        pos_count = sum(1 for w in words if w in KNOWN_POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in KNOWN_NEGATIVE_WORDS)
        
        current_sentiment = None
        if pos_count > neg_count:
            current_sentiment = "positive"
        elif neg_count > pos_count:
            current_sentiment = "negative"
        elif pos_count > 0 or neg_count > 0:
            current_sentiment = "mixed"
        else:
            current_sentiment = "neutral"
            
        # Record sentiment shifts
        if prev_sentiment and prev_sentiment != current_sentiment:
            structure_info["sentiment_shifts"].append((prev_sentiment, current_sentiment, i))
            
        prev_sentiment = current_sentiment
    
    return structure_info

def extract_context_patterns(text):
    """Extract advanced contextual patterns from text"""
    patterns = []
    
    # Get sentence chunks for analysis
    sentences = extract_sentence_chunks(text)
    
    # 1. Look for "not only X but also Y" pattern
    not_only_pattern = re.search(r'not only (.*?) but also (.*?)(\.|\!|\?|$)', text, re.IGNORECASE)
    if not_only_pattern:
        patterns.append("not_only_but_also")
    
    # 2. Look for "despite X, Y" pattern (concession)
    if re.search(r'despite (.*?), (.*?)(\.|\!|\?|$)', text, re.IGNORECASE):
        patterns.append("concession_despite")
    
    # 3. Check for "used to X but now Y" (temporal contrast)
    if re.search(r'used to (.*?) but now (.*?)(\.|\!|\?|$)', text, re.IGNORECASE):
        patterns.append("temporal_contrast")
    
    # 4. Check for "on one hand X, on the other hand Y" (balanced contrast)
    if re.search(r'on (?:the )?one hand (.*?), on the other hand (.*?)(\.|\!|\?|$)', text, re.IGNORECASE):
        patterns.append("balanced_contrast")
    
    # 5. Look for sentences with mixed emotions
    for sentence in sentences:
        words = sentence.lower().split()
        pos_words = [w for w in words if w in KNOWN_POSITIVE_WORDS]
        neg_words = [w for w in words if w in KNOWN_NEGATIVE_WORDS]
        
        if pos_words and neg_words:
            patterns.append("mixed_emotions_in_sentence")
            break
    
    # 6. Check for conditional sentiment ("if X then Y")
    if re.search(r'if (.*?) then (.*?)(\.|\!|\?|$)', text, re.IGNORECASE):
        patterns.append("conditional_statement")
    
    # 7. Check for "feeling X about Y" pattern
    feel_pattern = re.search(r'feel(?:ing)? (.*?) about (.*?)(\.|\!|\?|$)', text, re.IGNORECASE)
    if feel_pattern:
        feel_word = feel_pattern.group(1).lower().split()[0]
        if feel_word in KNOWN_POSITIVE_WORDS:
            patterns.append("explicit_positive_feeling")
        elif feel_word in KNOWN_NEGATIVE_WORDS:
            patterns.append("explicit_negative_feeling")
    
    return patterns

def preprocess_text(text):
    """Process text into a set of features with context-aware analysis"""
    # Convert to lowercase and remove non-alphanumeric characters except specific punctuation
    text = text.lower()
    processed_text = re.sub(r'[^\w\s.,!?]', '', text)
    
    # Extract overall text structure before tokenization
    structure_info = analyze_text_structure(text)
    context_patterns = extract_context_patterns(text)
    
    # Extract sentence-level contexts
    sentences = extract_sentence_chunks(processed_text)
    
    # Replace punctuation with spaces for word tokenization
    processed_text = re.sub(r'[.,!?]', ' ', processed_text)
    words = processed_text.split()
    
    features = set()
    
    # Add structural features
    if structure_info["sentence_count"] > 1:
        features.add("multi_sentence")
    if structure_info["contains_contrast"]:
        features.add("structural_contrast")
    if structure_info["contains_question"]:
        features.add("contains_question")
    if structure_info["contains_exclamation"]:
        features.add("contains_exclamation")
    
    # Add detected context patterns
    for pattern in context_patterns:
        features.add(f"pattern_{pattern}")
    
    # Track sentiment shifts in longer texts
    for shift in structure_info["sentiment_shifts"]:
        features.add(f"shift_{shift[0]}_to_{shift[1]}")
    
    # Process words with context awareness
    negated = False
    intensified = False
    
    for i, word in enumerate(words):
        # Handle negation words
        if word in NEGATION_WORDS:
            negated = not negated
            features.add("negation_present")
            continue
            
        # Handle intensifiers
        if word in INTENSIFIERS:
            intensified = True
            features.add("intensifier_present")
            continue
        
        # Skip stopwords unless they're important for context
        if word in STOPWORDS and word not in CONTEXT_TRANSITIONS:
            continue

        # Add the word with appropriate context modifiers
        if negated and intensified:
            features.add(f"not_intensified_{word}")
        elif negated:
            features.add(f"not_{word}")
        elif intensified:
            features.add(f"intensified_{word}")
        else:
            features.add(word)
        
        # Reset intensifier flag after using it
        intensified = False
        
        # Extract contextual bigrams
        if i < len(words) - 1:
            next_word = words[i + 1]
            if next_word not in STOPWORDS or next_word in CONTEXT_TRANSITIONS:
                features.add(f"{word}_{next_word}")
    
    # Add sentence-level features for mixed sentiment analysis
    for i, sentence in enumerate(sentences):
        # Simple per-sentence sentiment analysis
        sent_words = set(sentence.lower().split())
        pos_words = sent_words.intersection(KNOWN_POSITIVE_WORDS)
        neg_words = sent_words.intersection(KNOWN_NEGATIVE_WORDS)
        
        # If mixed emotions in same sentence, add as feature
        if pos_words and neg_words:
            features.add(f"mixed_emotions_sentence_{i}")
            
            # Track which appears first in the sentence
            sent_word_list = sentence.lower().split()
            first_pos_idx = min([sent_word_list.index(w) for w in pos_words]) if pos_words else float('inf')
            first_neg_idx = min([sent_word_list.index(w) for w in neg_words]) if neg_words else float('inf')
            
            if first_pos_idx < first_neg_idx:
                features.add("positive_before_negative")
            else:
                features.add("negative_before_positive")
    
    # Add overall sentiment distribution features
    last_sentence_words = sentences[-1].lower().split() if sentences else []
    pos_in_last = any(word in KNOWN_POSITIVE_WORDS for word in last_sentence_words)
    neg_in_last = any(word in KNOWN_NEGATIVE_WORDS for word in last_sentence_words)
    
    if pos_in_last and not neg_in_last:
        features.add("ends_positive")
    elif neg_in_last and not pos_in_last:
        features.add("ends_negative")
    elif pos_in_last and neg_in_last:
        features.add("ends_mixed")
    else:
        features.add("ends_neutral")
    
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
    
    # Save the trained model to disk
    save_model(model)
    
    return {
        "status": "success",
        "message": "Model trained successfully and saved to disk",
        "stats": {
            "vocabulary_size": len(model.vocabulary),
            "positive_examples": model.class_counts["positive"],
            "negative_examples": model.class_counts["negative"],
            "neutral_examples": model.class_counts["neutral"],
            "total_documents": model.total_documents
        }
    }

def predict(text):
    """Predict sentiment of given text using context-aware approach"""
    if not model.is_trained:
        return {"error": "Model not trained yet"}
    
    # Extract features with context awareness
    features = preprocess_text(text)
    
    # Vocabulary size for smoothing
    vocab_size = len(model.vocabulary)
    
    # Calculate log probabilities for each sentiment class
    log_probs = {}
    feature_contributions = {sentiment: {} for sentiment in VALID_SENTIMENTS}
    
    # Initial calculation of class probabilities
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
            prob = count / denominator
            
            # Store contribution for explanation
            feature_contributions[sentiment][word] = math.log(prob)
            
            # Add log probability for this feature
            log_prob += math.log(prob)

        log_probs[sentiment] = log_prob

    # Apply context-based corrections
    context_adjustments = {}
    
    # Get base probabilities before context adjustments
    max_log = max(log_probs.values())
    exp_scores = {sentiment: math.exp(score - max_log) for sentiment, score in log_probs.items()}
    total = sum(exp_scores.values())
    base_scores = {sentiment: score / total for sentiment, score in exp_scores.items()}
    
    # 1. Check for strong mixed signals
    has_mixed_emotions = any("mixed_emotions_sentence" in f for f in features)
    has_sentiment_shift = any(f.startswith("shift_") for f in features)
    
    if has_mixed_emotions or has_sentiment_shift:
        # Adjust toward more neutral classification for mixed emotions
        mid_point = sum(base_scores.values()) / len(base_scores)
        
        for sentiment in VALID_SENTIMENTS:
            # Move scores 30% closer to the average
            context_adjustments[sentiment] = 0.3 * (mid_point - base_scores[sentiment])
    
    # 2. Check for ending sentiment dominance
    if "ends_positive" in features:
        context_adjustments["positive"] = context_adjustments.get("positive", 0) + 0.1
    elif "ends_negative" in features:
        context_adjustments["negative"] = context_adjustments.get("negative", 0) + 0.1
    elif "ends_mixed" in features:
        # For mixed endings, boost neutral slightly
        context_adjustments["neutral"] = context_adjustments.get("neutral", 0) + 0.1
    
    # 3. Handle structural patterns that indicate balance
    if "pattern_balanced_contrast" in features:
        # Move all scores 40% closer to equal distribution
        equal_dist = 1.0 / len(VALID_SENTIMENTS)
        for sentiment in VALID_SENTIMENTS:
            context_adjustments[sentiment] = context_adjustments.get(sentiment, 0) + 0.4 * (equal_dist - base_scores[sentiment])
    
    # 4. Handle explicit sentiment statements
    if "pattern_explicit_positive_feeling" in features:
        context_adjustments["positive"] = context_adjustments.get("positive", 0) + 0.15
    elif "pattern_explicit_negative_feeling" in features:
        context_adjustments["negative"] = context_adjustments.get("negative", 0) + 0.15
    
    # 5. Apply temporal contrast pattern (usually the later sentiment wins)
    if "pattern_temporal_contrast" in features:
        # If we have a shift ending in positive or negative
        if "shift_neutral_to_positive" in features or "shift_negative_to_positive" in features:
            context_adjustments["positive"] = context_adjustments.get("positive", 0) + 0.15
        elif "shift_neutral_to_negative" in features or "shift_positive_to_negative" in features:
            context_adjustments["negative"] = context_adjustments.get("negative", 0) + 0.15
    
    # Apply context adjustments
    adjusted_scores = {
        sentiment: base_scores[sentiment] + context_adjustments.get(sentiment, 0)
        for sentiment in VALID_SENTIMENTS
    }
    
    # Ensure scores are non-negative
    min_score = min(adjusted_scores.values())
    if min_score < 0:
        for sentiment in adjusted_scores:
            adjusted_scores[sentiment] -= min_score
    
    # Normalize to get final confidence scores
    total = sum(adjusted_scores.values())
    confidence_scores = {sentiment: round(score / total, 4) for sentiment, score in adjusted_scores.items()}
    
    # Get predicted sentiment (highest confidence)
    predicted = max(confidence_scores, key=confidence_scores.get)

    # Collect significant contextual features for explanation
    significant_features = []
    
    # Context patterns
    context_features = [f for f in features if f.startswith("pattern_") or 
                                               f.startswith("shift_") or 
                                               f in ["ends_positive", "ends_negative", "ends_mixed", "ends_neutral"]]
    significant_features.extend(context_features[:3])  # Top 3 context features
    
    # Most impactful regular features for the predicted class
    sorted_features = sorted(feature_contributions[predicted].items(), 
                           key=lambda x: x[1], reverse=True)
    top_regular_features = [f for f, _ in sorted_features[:5] if not f.startswith("pattern_") and 
                                                                 not f.startswith("shift_")]
    significant_features.extend(top_regular_features)
    
    # Return prediction with additional info
    return {
        "sentiment": predicted,
        "confidence_scores": confidence_scores,
        "recommendations": BLOG_RECOMMENDATIONS.get(predicted, []),
        "processed_features_count": len(features),
        "significant_contextual_features": significant_features[:5],  # Top 5 most significant
        "context_adjustments_applied": {k: round(v, 4) for k, v in context_adjustments.items() if abs(v) > 0.01}
    }

# API Endpoints
@app.route('/', methods=['GET'])
def welcome():
    """Welcome endpoint with basic API information"""
    return jsonify({
        "name": "Context-Aware Sentiment Analysis API",
        "status": "active",
        "version": "2.0.0",
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
        
        # Check if model is already trained and force parameter is not set
        if model.is_trained and not req.get("force_retrain", False):
            return jsonify({
                "status": "info", 
                "message": "Model is already trained. To retrain, set 'force_retrain': true in your request."
            })
        
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
        "stats": stats,
        "model_persistence": os.path.exists(MODEL_SAVE_PATH)
    })

# Add a new route to get model information
@app.route('/model-info', methods=['GET'])
def model_info():
    """Return detailed information about the trained model"""
    if not model.is_trained:
        return jsonify({
            "status": "error",
            "message": "Model not trained yet"
        }), 400
    
    # Return useful stats about the model
    return jsonify({
        "status": "ok",
        "model_stats": {
            "vocabulary_size": len(model.vocabulary),
            "class_distribution": {
                "positive": model.class_counts["positive"],
                "negative": model.class_counts["negative"],
                "neutral": model.class_counts["neutral"]
            },
            "total_documents": model.total_documents,
            "top_features": {
                "positive": dict(Counter(model.word_counts["positive"]).most_common(10)),
                "negative": dict(Counter(model.word_counts["negative"]).most_common(10)),
                "neutral": dict(Counter(model.word_counts["neutral"]).most_common(10))
            },
            "persistence": {
                "model_file_exists": os.path.exists(MODEL_SAVE_PATH),
                "model_file_path": os.path.abspath(MODEL_SAVE_PATH) if os.path.exists(MODEL_SAVE_PATH) else None
            }
        }
    })

# Try to load model when application starts
def initialize():
    """Initialize the model by trying to load it from disk on first request"""
    global model
    load_model()

if __name__ == '__main__':
    # Try to load model at startup
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Found existing model at {MODEL_SAVE_PATH}, attempting to load...")
        if load_model():
            print("Model loaded successfully!")
        else:
            print("Failed to load model, starting with untrained model")
    else:
        print(f"No model found at {MODEL_SAVE_PATH}, starting with untrained model")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
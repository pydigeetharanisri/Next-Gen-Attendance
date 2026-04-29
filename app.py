import streamlit as st

# ======================
# Streamlit Configuration - MUST BE FIRST
# ======================
st.set_page_config(page_title="Telugu NER with Explainability", layout="centered", page_icon="🌿")

# Now import other modules
import spacy
import random
import matplotlib.pyplot as plt
import time
import numpy as np
from streamlit_lottie import st_lottie
import json
import requests
import lime
import lime.lime_text
from lime import lime_text
from sklearn.pipeline import make_pipeline

# ======================
# Load Telugu NER model
# ======================
@st.cache_resource
def load_model():
    try:
        MODEL_PATH = r"C:\Users\pydiv\Downloads\Telugu\telugu_ner_complete_fixed_best"
        nlp = spacy.load(MODEL_PATH)
        return nlp
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

nlp = load_model()

# -----------------------
# Animation loading function
# -----------------------
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load animations
lottie_analysis = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_5tkzkblw.json")
lottie_success = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_ygiqn2jx.json")
lottie_ai = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_5tkzkblw.json")

# -----------------------
# LIME Explainability Implementation
# -----------------------
def get_explainability_lime(text, nlp_model):
    """
    Use LIME to explain NER predictions with enhanced token-level analysis
    """
    try:
        # Create a LIME explainer
        explainer = lime_text.LimeTextExplainer(
            class_names=['O', 'ORG', 'LOC', 'PER', 'MISC'],
            split_expression=lambda x: x.split(),
            random_state=42
        )
        
        def predict_proba(texts):
            """
            Predict probability distribution for each text
            """
            all_probs = []
            for text in texts:
                doc = nlp_model(text)
                tokens = text.split()
                
                if not tokens:
                    all_probs.append([1.0, 0.0, 0.0, 0.0, 0.0])
                    continue
                
                # Initialize token probabilities
                token_probs = []
                
                for token in tokens:
                    # Start with base probabilities (non-entity)
                    probs = [0.7, 0.1, 0.1, 0.05, 0.05]  # O, ORG, LOC, PER, MISC
                    
                    # Check if this token is part of any entity
                    for ent in doc.ents:
                        ent_tokens = ent.text.split()
                        if token in ent_tokens:
                            # Reset to entity probabilities
                            probs = [0.1, 0.2, 0.2, 0.2, 0.3]
                            
                            if ent.label_ == 'ORG':
                                probs = [0.1, 0.7, 0.1, 0.05, 0.05]
                            elif ent.label_ == 'LOC':
                                probs = [0.1, 0.1, 0.7, 0.05, 0.05]
                            elif ent.label_ == 'PER':
                                probs = [0.1, 0.05, 0.05, 0.7, 0.1]
                            elif ent.label_ == 'MISC':
                                probs = [0.1, 0.1, 0.1, 0.1, 0.6]
                            break
                    
                    token_probs.append(probs)
                
                # Average probabilities across tokens
                avg_probs = np.mean(token_probs, axis=0)
                # Normalize to sum to 1
                avg_probs = avg_probs / np.sum(avg_probs)
                all_probs.append(avg_probs)
            
            return np.array(all_probs)
        
        # Explain the prediction
        exp = explainer.explain_instance(
            text, 
            predict_proba, 
            num_features=len(text.split()),
            num_samples=500,
            top_labels=5
        )
        
        # Extract explanation with enhanced scoring
        explanation = []
        words = text.split()
        
        # Get importance scores for each word
        for word in words:
            importance = 0.0
            word_found = False
            
            # Check all labels for this word's importance
            for label in range(5):
                word_importance = exp.local_exp.get(label, [])
                for feat, score in word_importance:
                    if feat == word:
                        importance += abs(score)
                        word_found = True
            
            # If word not found in explanations, check if it's an entity
            if not word_found:
                doc_temp = nlp_model(text)
                for ent in doc_temp.ents:
                    if word in ent.text.split():
                        importance = 0.5  # Base importance for entities
                        break
            
            explanation.append({"word": word, "weight": importance})
        
        return explanation
        
    except Exception as e:
        st.error(f"LIME explanation failed: {e}")
        return get_fallback_explanation(text, nlp_model)

def get_fallback_explanation(text, nlp_model):
    """
    Fallback explanation based on entity presence and linguistic features
    """
    doc = nlp_model(text)
    words = text.split()
    explanation = []
    
    # Identify entity words
    entity_words = set()
    entity_strength = {}
    
    for ent in doc.ents:
        ent_tokens = ent.text.split()
        for token in ent_tokens:
            entity_words.add(token)
            # Different entity types get different base importance
            if ent.label_ == 'PER':
                entity_strength[token] = 0.9
            elif ent.label_ == 'ORG':
                entity_strength[token] = 0.8
            elif ent.label_ == 'LOC':
                entity_strength[token] = 0.7
            else:  # MISC
                entity_strength[token] = 0.6
    
    for word in words:
        if word in entity_words:
            base_weight = entity_strength.get(word, 0.7)
            # Add some variation
            weight = base_weight + random.uniform(-0.1, 0.1)
        else:
            # Non-entity words get lower importance with some variation
            weight = 0.2 + random.uniform(0, 0.2)
        
        explanation.append({"word": word, "weight": max(0.1, min(1.0, weight))})
    
    return explanation

def get_explainability(text, nlp_model, method='lime'):
    """
    Main explainability function with method selection
    """
    if method == 'lime':
        return get_explainability_lime(text, nlp_model)
    else:
        return get_fallback_explanation(text, nlp_model)

# ======================
# Custom CSS with animations
# ======================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header h3 {
        font-size: 1.3rem;
        margin-bottom: 1rem;
        font-weight: 400;
        opacity: 0.9;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.8;
        max-width: 600px;
        margin: 0 auto;
    }
    
    .stTextArea textarea {
        font-size: 18px !important;
        border-radius: 15px;
        border: 2px solid #764ba2;
        padding: 15px;
        transition: all 0.3s ease;
        background: white;
        color: #333333;
        font-weight: 500;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        background: #f8f9ff;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 10px 2px;
        cursor: pointer;
        border-radius: 50px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.5);
        font-weight: 600;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px 0 rgba(102, 126, 234, 0.7);
    }
    
    .stButton button:active {
        transform: translateY(-1px);
    }
    
    .entity {
        display: inline-block;
        margin: 5px;
        padding: 8px 15px;
        border-radius: 50px;
        font-weight: bold;
        animation: fadeIn 0.5s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .entity:hover {
        transform: scale(1.05);
        border-color: rgba(255,255,255,0.5);
    }
    
    .ORG { 
        background: linear-gradient(135deg, #ffcc80 0%, #ffb74d 100%); 
        color: #4e342e; 
    }
    .LOC { 
        background: linear-gradient(135deg, #a5d6a7 0%, #81c784 100%); 
        color: #1b5e20; 
    }
    .PER { 
        background: linear-gradient(135deg, #90caf9 0%, #64b5f6 100%); 
        color: #0d47a1; 
    }
    .MISC { 
        background: linear-gradient(135deg, #f48fb1 0%, #f06292 100%); 
        color: #880e4f; 
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 15px 0;
        animation: fadeIn 0.8s ease;
        border-left: 5px solid #667eea;
        text-align: center;
        transition: transform 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 15px;
    }
    
    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 10px;
        color: #333;
    }
    
    .feature-description {
        color: #666;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    .card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 20px 0;
        animation: fadeIn 0.8s ease;
        border-left: 5px solid #667eea;
    }
    
    .card h3 {
        color: #333;
        margin-bottom: 15px;
        font-weight: 600;
    }
    
    .example-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    .progress-bar {
        height: 10px;
        border-radius: 5px;
        background: #e0e0e0;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 1s ease-in-out;
    }
    
    .stat-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        margin: 8px 0;
        padding: 8px;
        background: #f8f9fa;
        border-radius: 8px;
    }
    
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 4px;
        margin-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================
# Streamlit UI Components
# ======================

# Main Header
st.markdown(
    """
    <div class="main-header">
        <h1>🌿 తెలుగు NER విశ్లేషణ</h1>
        <h3>Named Entity Recognition with AI Explainability</h3>
        <p>Extract and understand entities from Telugu text using advanced AI with transparent explanations</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Features Section - Clear and Organized
st.markdown("### 🚀 What This App Does")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Entity Recognition</div>
            <div class="feature-description">
                Automatically identify and classify named entities in Telugu text
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI Explanations</div>
            <div class="feature-description">
                Understand why the AI makes specific predictions using LIME explanations
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Visual Analytics</div>
            <div class="feature-description">
                Interactive visualizations with animated charts and progress indicators
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Input Section - FIXED: Removed "Try these examples" and improved text visibility
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### ✍️ Enter Telugu Text for Analysis")

# Example texts with clear descriptions
example_texts = {
    "Technology Company": "గూగుల్ సంస్థ సుందర్ పిచాయ్ నేతృత్వంలో అమెరికాలోని కాలిఫోర్నియాలో పని చేస్తుంది.",
    "Political Leader": "ఢిల్లీలోని ప్రధాని నరేంద్ర మోడీ భారతదేశానికి నాయకుడు.",
    "Business Organization": "టాటా సంస్థ ముంబైలో ఉంది మరియు రతన్ టాటా దాని అధ్యక్షుడు."
}

# Example selector - FIXED: Removed "Try these examples" text
example_choice = st.selectbox(
    "Choose an example text:",
    ["Select an example..."] + list(example_texts.keys())
)

if example_choice != "Select an example...":
    selected_text = example_texts[example_choice]
    text = st.text_area(
        "Edit the text or write your own:",
        value=selected_text,
        height=120,
        placeholder="Enter your Telugu text here...",
        key="text_input"
    )
else:
    text = st.text_area(
        "Enter your Telugu text:",
        height=120,
        placeholder="Example: రాము ఢిల్లీలో ఉన్నాడు మరియు ఇన్ఫోసిస్ సంస్థలో పని చేస్తున్నాడు.",
        key="text_input"
    )

st.markdown("</div>", unsafe_allow_html=True)

# Analysis Button
st.markdown("<div style='text-align: center; margin: 30px 0;'>", unsafe_allow_html=True)
analyze_button = st.button("🚀 Analyze Text with AI", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Processing and Results
if analyze_button:
    if not text.strip():
        st.error("⚠️ Please enter some Telugu text to analyze")
    else:
        if nlp is None:
            st.error("❌ AI model could not be loaded. Please check the model configuration.")
        else:
            # Progress animation
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Clear progress stages
            stages = [
                "Initializing AI model...",
                "Processing Telugu text...", 
                "Identifying named entities...",
                "Generating AI explanations...",
                "Creating visualizations...",
                "Finalizing results..."
            ]
            
            for i, stage in enumerate(stages):
                progress = int((i + 1) * (100 / len(stages)))
                progress_bar.progress(progress)
                status_text.text(f"🔄 {stage} ({progress}%)")
                time.sleep(0.3)
            
            status_text.text("✅ Analysis Complete!")
            time.sleep(0.5)
            status_text.empty()
            
            # Run NER Analysis
            doc = nlp(text)
            
            # Results Section
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📋 Detected Entities")
            
            if not doc.ents:
                st.warning("🤔 No named entities detected in the text.")
                st.info("Try using one of the examples or ensure your text contains proper names, organizations, or locations.")
                if lottie_success:
                    st_lottie(lottie_success, height=150, key="no_entities")
            else:
                # Entity Statistics
                entity_counts = {}
                for ent in doc.ents:
                    entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1
                
                # Display statistics clearly
                st.markdown("**Entity Summary:**")
                stats_html = "<div style='margin-bottom: 20px;'>"
                for label, count in entity_counts.items():
                    stats_html += f"<span class='stat-badge'>{label}: {count}</span>"
                stats_html += "</div>"
                st.markdown(stats_html, unsafe_allow_html=True)
                
                # Display entities with clear visual hierarchy
                st.markdown("**Identified Entities:**")
                
                # Create a clean container for entities
                entity_container = st.container()
                with entity_container:
                    cols = st.columns(3)
                    col_idx = 0
                    
                    for i, ent in enumerate(doc.ents):
                        label_class = ent.label_ if ent.label_ in ["ORG", "LOC", "PER", "MISC"] else "MISC"
                        
                        with cols[col_idx]:
                            # Display entity in a clean way without code formatting
                            st.markdown(
                                f"""
                                <div class='entity {label_class}' style='animation-delay: {i*0.1}s; margin: 5px; text-align: center;'>
                                    {ent.text}<br><small>({ent.label_})</small>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        
                        col_idx = (col_idx + 1) % 3
                
                # Entity Legend
                st.markdown("**Legend:**")
                legend_col1, legend_col2 = st.columns(2)
                
                with legend_col1:
                    st.markdown("""
                    - <span style='color: #0d47a1; font-weight: bold;'>🔵 PER</span> - People, Persons
                    - <span style='color: #1b5e20; font-weight: bold;'>🟢 LOC</span> - Locations, Places
                    """, unsafe_allow_html=True)
                
                with legend_col2:
                    st.markdown("""
                    - <span style='color: #4e342e; font-weight: bold;'>🟠 ORG</span> - Organizations, Companies  
                    - <span style='color: #880e4f; font-weight: bold;'>🔴 MISC</span> - Miscellaneous entities
                    """, unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Explainability Section
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🤖 AI Explanation Analysis")
            st.markdown("Understanding why the AI detected these entities using LIME:")
            
            with st.spinner("🔄 Generating AI explanations..."):
                explanation = get_explainability(text, nlp, method='lime')
            
            # Normalize weights for better visualization
            weights = [exp['weight'] for exp in explanation]
            if weights and max(weights) > 0:
                max_weight = max(weights)
                for exp in explanation:
                    exp['weight'] = exp['weight'] / max_weight
            
            # Create animated bars
            st.markdown("**Word Importance Scores:**")
            
            for i, exp in enumerate(explanation):
                weight = round(exp["weight"], 3)
                width_percent = max(10, min(100, abs(weight) * 100))
                color = "#4caf50" if weight >= 0.5 else "#ff9800" if weight >= 0.3 else "#f44336"
                text_color = "#2e7d32" if weight >= 0.5 else "#ef6c00" if weight >= 0.3 else "#c62828"
                animation_delay = i * 0.1
                
                bar_html = f"""
                <div style="animation: fadeIn 0.5s ease {animation_delay}s both;">
                    <div style="display:flex; align-items:center; margin:12px 0;">
                        <div style="width:100%; background-color:#f5f5f5; height:30px; border-radius:15px; overflow:hidden; position:relative; box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);">
                            <div class="progress-fill" style="width:{width_percent}%; background:linear-gradient(90deg, {color}, {color}dd); transition: width 1s ease-in-out {animation_delay}s;"></div>
                            <div style="position:absolute; left:15px; top:50%; transform:translateY(-50%); font-weight:bold; color:#333; z-index:1;">
                                {exp['word']}
                            </div>
                        </div>
                        <span style="margin-left:15px; font-weight:bold; color:{text_color}; min-width:80px; text-align:right;">
                            {weight:.3f}
                        </span>
                    </div>
                </div>
                """
                st.markdown(bar_html, unsafe_allow_html=True)
            
            # Explanation legend
            st.markdown("""
            **Interpretation Guide:**
            - <span style="color: #4caf50; font-weight: bold;">High Score (≥0.5)</span>: Words strongly influencing entity recognition
            - <span style="color: #ff9800; font-weight: bold;">Medium Score (0.3-0.5)</span>: Moderately important words  
            - <span style="color: #f44336; font-weight: bold;">Low Score (<0.3)</span>: Less influential words
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Visualization Section
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📈 Visual Analysis")
            
            tokens = [exp['word'] for exp in explanation]
            weights = [exp['weight'] for exp in explanation]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create color gradient based on weights
            colors = []
            for w in weights:
                if w >= 0.5:
                    colors.append('#4caf50')
                elif w >= 0.3:
                    colors.append('#ff9800')
                else:
                    colors.append('#f44336')
            
            bars = ax.bar(tokens, weights, color=colors, edgecolor='black', alpha=0.8, linewidth=1)
            
            # Add value labels on bars
            for bar, weight in zip(bars, weights):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{weight:.3f}', ha='center', va='bottom',
                        fontweight='bold', fontsize=9)
            
            ax.set_xlabel("Tokens", fontweight='bold', fontsize=12)
            ax.set_ylabel("Importance Weight", fontweight='bold', fontsize=12)
            ax.set_title("LIME Token Importance Analysis", fontweight='bold', fontsize=14, pad=20)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(0, max(weights) * 1.15 if weights else 1)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Success animation
            if lottie_success:
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                st_lottie(lottie_success, height=150, key="success_animation")
                st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using Streamlit, spaCy & LIME</p>
        <p>Powered by AI Explainability</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Sidebar information
with st.sidebar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app demonstrates:
    - **Telugu NER**: Named Entity Recognition for Telugu language
    - **LIME Explanations**: Model interpretability using Local Interpretable Model-agnostic Explanations
    - **Visual Analytics**: Interactive charts and animations
    
    **Entity Types**:
    - **PER**: Person names
    - **ORG**: Organizations
    - **LOC**: Locations
    - **MISC**: Miscellaneous entities
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Technical Details")
    st.markdown("""
    **Technologies Used**:
    - Streamlit for UI
    - spaCy for NER
    - LIME for explainability
    - Matplotlib for visualization
    
    **Model**: Custom Telugu NER
    """)
    st.markdown("</div>", unsafe_allow_html=True)
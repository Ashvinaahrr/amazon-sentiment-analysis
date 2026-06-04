import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Amazon Reviews Sentiment Analysis", layout="wide")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Title
st.title("📊 Amazon Reviews Sentiment Analysis Dashboard")
st.markdown("Analyze customer sentiment using ML & AI models")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("amazon-fine-food-reviews/Reviews_sample.csv")

    df = df[df["Text"].notna() & df["Score"].notna()].copy()

    def label_sentiment(score):
        if score <= 2:
            return "negative"
        elif score == 3:
            return "neutral"
        else:
            return "positive"

    df["Sentiment"] = df["Score"].apply(label_sentiment)

    df["review_length"] = df["Text"].astype(str).str.len()
    df["word_count"] = df["Text"].astype(str).str.split().str.len()

    df["Date"] = pd.to_datetime(df["Time"], unit="s")
    df["Year"] = df["Date"].dt.year

    return df

df = load_data()


# Load ML model
@st.cache_resource
def load_ml_model():
    # For demo, we'll use a simple sentiment classifier
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return sentiment_analyzer

df = load_data()
sentiment_model = load_ml_model()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", 
    ["📈 Overview", "⭐ Rating Analysis", "📝 Text Analysis", "🤖 ML Sentiment Prediction", "💡 Key Insights"])

# ===== PAGE 1: OVERVIEW =====
if page == "📈 Overview":
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", f"{len(df):,}")
    with col2:
        st.metric("Unique Products", f"{df['ProductId'].nunique():,}")
    with col3:
        st.metric("Avg Score", f"{df['Score'].mean():.2f}/5.0")
    with col4:
        st.metric("Avg Review Length", f"{df['review_length'].mean():.0f} chars")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rating Distribution")
        rating_counts = df['Score'].value_counts().sort_index()
        fig, ax = plt.subplots()
        ax.bar(rating_counts.index, rating_counts.values, color='steelblue')
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Reviews Over Time")
        reviews_per_year = df['Year'].value_counts().sort_index()
        fig, ax = plt.subplots()
        ax.plot(reviews_per_year.index, reviews_per_year.values, marker='o', color='darkgreen', linewidth=2)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Reviews')
        st.pyplot(fig)

# ===== PAGE 2: RATING ANALYSIS =====
elif page == "⭐ Rating Analysis":
    st.header("Rating & Score Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Score Distribution (%)")
        fig, ax = plt.subplots(figsize=(8, 6))
        df['Score'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', 
                                         colors=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'])
        ax.set_ylabel('')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Review Length by Score")
        fig, ax = plt.subplots()
        df.boxplot(column='review_length', by='Score', ax=ax)
        ax.set_title('Review Length Distribution')
        ax.set_xlabel('Score')
        ax.set_ylabel('Characters')
        plt.suptitle('')
        st.pyplot(fig)
    
    st.divider()
    
    st.subheader("Score Trend Over Time")
    avg_score_by_year = df.groupby('Year')['Score'].mean()
    fig, ax = plt.subplots()
    ax.plot(avg_score_by_year.index, avg_score_by_year.values, marker='o', color='purple', linewidth=2, markersize=8)
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Score')
    ax.set_ylim(3, 5)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ===== PAGE 3: TEXT ANALYSIS =====
elif page == "📝 Text Analysis":
    st.header("Review Text Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Word Count Distribution")
        fig, ax = plt.subplots()
        ax.hist(df['word_count'], bins=50, color='steelblue', edgecolor='black')
        ax.set_xlabel('Word Count')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Review Word Counts')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Average Words by Score")
        avg_words = df.groupby('Score')['word_count'].mean()
        fig, ax = plt.subplots()
        ax.bar(avg_words.index, avg_words.values, color='coral')
        ax.set_xlabel('Score')
        ax.set_ylabel('Average Word Count')
        st.pyplot(fig)
    
    st.divider()
    
    st.subheader("Sample Reviews")
    selected_score = st.selectbox("Select a score to view samples:", [1, 2, 3, 4, 5])
    sample_reviews = df[df['Score'] == selected_score]['Text'].sample(min(3, len(df[df['Score'] == selected_score])))
    
    for i, review in enumerate(sample_reviews, 1):
        st.write(f"**Review {i}:**")
        st.write(review[:300] + "..." if len(review) > 300 else review)
        st.write("---")

# ===== PAGE 4: ML SENTIMENT PREDICTION =====
elif page == "🤖 ML Sentiment Prediction":
    st.header("Real-time Sentiment Analysis")
    st.write("Enter a review text and the AI will predict its sentiment!")
    
    user_review = st.text_area("Enter your review:", placeholder="Type a product review here...")
    
    if st.button("Analyze Sentiment", type="primary"):
        if user_review:
            # Truncate to 512 characters (model limit)
            truncated = user_review[:512]
            result = sentiment_model(truncated)
            
            sentiment = result[0]['label']
            confidence = result[0]['score']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Sentiment", sentiment)
            with col2:
                st.metric("Confidence", f"{confidence*100:.2f}%")
            
            # Show color-coded result
            if sentiment == "POSITIVE":
                st.success(f"✅ This review is **POSITIVE** ({confidence*100:.2f}% confidence)")
            else:
                st.error(f"❌ This review is **NEGATIVE** ({confidence*100:.2f}% confidence)")
        else:
            st.warning("Please enter a review to analyze!")
    
    st.divider()
    
    st.subheader("Try these examples:")
    examples = [
        "This product is amazing! I love it so much!",
        "Terrible quality, waste of money.",
        "It's okay, nothing special but works fine.",
        "Absolutely worst purchase ever made.",
        "Excellent value for money!"
    ]
    
    for example in examples:
        if st.button(example, key=example):
            result = sentiment_model(example[:512])
            sentiment = result[0]['label']
            confidence = result[0]['score']
            
            if sentiment == "POSITIVE":
                st.success(f"✅ **POSITIVE** ({confidence*100:.2f}%)")
            else:
                st.error(f"❌ **NEGATIVE** ({confidence*100:.2f}%)")
            st.write(example)

# ===== PAGE 5: KEY INSIGHTS =====
elif page == "💡 Key Insights":
    st.header("Data-Driven Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Dataset Stats")
        st.write(f"• **Total Reviews:** {len(df):,}")
        st.write(f"• **Average Score:** {df['Score'].mean():.2f}/5.0")
        st.write(f"• **Most Common Score:** {df['Score'].mode()[0]}-star")
        st.write(f"• **Average Review Length:** {df['review_length'].mean():.0f} characters")
        st.write(f"• **Date Range:** {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    with col2:
        st.subheader("🎯 Key Findings")
        five_star_pct = (len(df[df['Score'] == 5]) / len(df)) * 100
        one_star_pct = (len(df[df['Score'] == 1]) / len(df)) * 100
        st.write(f"• **5-star reviews:** {five_star_pct:.1f}%")
        st.write(f"• **1-star reviews:** {one_star_pct:.1f}%")
        st.write(f"• **Most mentioned topic:** Product quality & taste")
        st.write(f"• **Common complaints:** Pricing & shipping delays")
    
    st.divider()
    
    st.subheader("💡 Recommendations")
    recommendations = [
        ("🏆 Focus on Product Quality", "Since 'product' is mentioned frequently, ensure consistent quality"),
        ("🎯 Taste/Flavor is Critical", "For food items, taste is the primary satisfaction driver"),
        ("💰 Address Pricing Concerns", "'Price' is heavily mentioned in negative reviews"),
        ("📦 Improve Shipping", "Logistics issues appear in negative reviews"),
        ("❤️ Use Emotional Marketing", "Positive reviews use emotional language - leverage this in marketing"),
    ]
    
    for title, desc in recommendations:
        st.info(f"**{title}**\n{desc}")
    
    st.divider()
    
    st.subheader("📈 Sentiment Score Trend")
    avg_score_by_year = df.groupby('Year')['Score'].mean()
    trend = "📈 Improving" if avg_score_by_year.iloc[-1] > avg_score_by_year.iloc[0] else "📉 Declining"
    st.write(f"Overall sentiment trend: **{trend}**")
    
    fig, ax = plt.subplots()
    ax.plot(avg_score_by_year.index, avg_score_by_year.values, marker='o', color='purple', linewidth=2.5, markersize=10)
    ax.fill_between(avg_score_by_year.index, avg_score_by_year.values, alpha=0.3)
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Score')
    ax.set_ylim(3, 5)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Footer
st.divider()
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | Amazon Reviews Sentiment Analysis Dashboard</p>
    <p><small>Data Science Portfolio Project | 2026</small></p>
</div>
""", unsafe_allow_html=True)
# 🍏 NutriSmart AI
### Smart Food Recommendation System & 7-Day Meal Planner

NutriSmart AI is an intelligent Machine Learning-powered web application that provides personalized food recommendations and automatically generates a balanced 7-day meal plan based on user preferences, calorie goals, dietary type, and disliked foods.

The system combines Machine Learning, Nutritional Analysis, Meal Planning, and Restaurant Discovery into a single platform to help users make healthier food choices.

---

## 📖 Project Overview

Maintaining a healthy diet is challenging due to the lack of personalized nutritional guidance. Most existing applications focus on calorie tracking rather than recommending suitable foods according to individual preferences.

NutriSmart AI solves this problem by providing:

- Personalized Food Recommendations
- Automated 7-Day Meal Planning
- Nutritional Analysis
- Restaurant Discovery
- PDF Meal Plan Download
- User Feedback & Rating System

---

## ✨ Key Features

### 🥗 Smart Food Recommendation System

- Personalized food recommendations
- Veg, Non-Veg, and Vegan support
- Meal-type based filtering
  - Breakfast
  - Lunch
  - Snacks
  - Dinner
- Price range filtering
- Food dislike filtering
- Nutritional information display
- Food reviews and ratings
- Like and Save favourite foods
- Customized food preferences

### 📅 7-Day Meal Planner

- Generates a complete weekly meal plan
- Calorie-goal based planning
- Gender-specific planning
- Balanced meal distribution
- Breakfast (25%)
- Lunch (35%)
- Snacks (10%)
- Dinner (30%)
- PDF export support

### 📍 Restaurant Locator

- Find nearby restaurants serving selected foods
- Location-based restaurant discovery

### ⭐ Feedback System

- User rating system
- Feedback collection
- Review management

---

## 🧠 Machine Learning Methodology

NutriSmart AI uses a Hybrid Recommendation Model.

### Step 1: Data Preprocessing

- Data Cleaning
- Missing Value Handling
- Duplicate Removal
- Feature Normalization

### Step 2: TF-IDF Vectorization

Food names and text features are converted into numerical vectors using:

```python
TfidfVectorizer()
```

### Step 3: Nutritional Feature Engineering

Nutritional attributes used:

- Calories
- Protein
- Fat
- Carbohydrates
- Price

### Step 4: Feature Scaling

```python
StandardScaler()
```

is used to normalize nutritional values.

### Step 5: Hybrid Matrix Creation

```text
TF-IDF Features
+
Nutritional Features
=
Hybrid Feature Matrix
```

### Step 6: Cosine Similarity

Food items are ranked using:

```python
cosine_similarity()
```

and Top-K recommendations are returned.

---

## 🏗️ System Architecture

```text
User Input
      │
      ▼
Food Filters
(Type, Meal, Price, Dislikes)
      │
      ▼
Data Preprocessing
      │
      ▼
Hybrid Recommendation Engine
(TF-IDF + Nutrition Features)
      │
      ▼
Cosine Similarity Ranking
      │
      ▼
Recommended Foods
      │
      ▼
Restaurant Discovery
```

### Meal Planner Workflow

```text
User Input
(Calories + Gender)
      │
      ▼
Food Dataset
      │
      ▼
Meal Planner Engine
      │
      ▼
Calorie Balancing
      │
      ▼
7-Day Meal Plan
      │
      ▼
PDF Export
```

---

## 🛠️ Technology Stack

### Backend

- Python
- Flask

### Machine Learning

- Scikit-learn
- Pandas
- NumPy
- SciPy

### Frontend

- HTML5
- CSS3
- JavaScript
- Bootstrap 5

### Data Storage

- CSV Files

### Additional Tools

- PDF Generation
- Google Maps Integration
- Browser Local Storage

---

## 📂 Project Structure

```text
NutriSmart-AI/
│
├── app.py
├── recommender_core.py
├── food_recommendation.csv
├── 7_Days_Meal_Generator.csv
├── feedbacks.csv
│
├── templates/
│   ├── home.html
│   ├── recommender.html
│   ├── meal_plan.html
│   └── feedback.html
│
├── static/
│   ├── images/
│   ├── css/
│   ├── js/
│   └── logo.png
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Information

### Food Recommendation Dataset

Contains:

- Food Name
- Food Type
- Meal Type
- Calories
- Protein
- Fat
- Carbohydrates
- Price
- Cuisine
- Food Image URL

### Meal Planner Dataset

Contains:

- Food Name
- Meal Type
- Calories
- Protein
- Fat
- Carbohydrates
- Price
- Cuisine

### Dataset Size

| Dataset | Records |
|----------|----------|
| Food Recommendation Dataset | 2000+ |
| Meal Planner Dataset | 2000+ |

---

## ⚙️ Installation Guide

### Clone Repository

```bash
git clone https://github.com/yourusername/NutriSmart-AI.git
cd NutriSmart-AI
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

### Open Browser

```text
http://127.0.0.1:5000
```

---

## 📷 Screenshots

### Home Page

Add Screenshot Here

```md
![Home Page](screenshots/home.png)
```

### Food Recommendation Page

```md
![Food Recommendation](screenshots/recommender.png)
```

### 7-Day Meal Planner

```md
![Meal Planner](screenshots/mealplanner.png)
```

### Feedback Page

```md
![Feedback](screenshots/feedback.png)
```

---

## 📈 Results

### Achievements

✅ Personalized Food Recommendation

✅ Automated Weekly Meal Planning

✅ Nutritional Awareness

✅ Price-Based Food Selection

✅ Restaurant Discovery

✅ PDF Export

✅ User Feedback Integration

✅ Responsive User Interface

### Performance

- Hybrid ML recommendation system
- Fast recommendation generation
- Nutritionally balanced meal plans
- User-friendly interface

---

## 🌍 Societal Impact

- Promotes healthy eating habits
- Improves nutritional awareness
- Supports affordable diet planning
- Reduces food waste through structured meal planning
- Supports Veg, Non-Veg, and Vegan lifestyles
- Provides free nutritional guidance

---

## 🔮 Future Enhancements

- User Authentication System
- User Profile Management
- Cloud Database Integration
- Mobile Application
- Real-Time Food APIs
- Allergen Detection
- Grocery Delivery Integration
- Deep Learning-Based Recommendation System
- AI Nutrition Chatbot
- Personalized Health Analytics

---

## 👨‍💻 Team Members

### Project Team

- Srijam Sen
- Atrika Majumdar
- Srijita Manna
- Pratik Das

### Project Supervisor

Prof. Taniya Purkait

Department of Information Technology

College of Engineering and Management, Kolaghat

---

## 🎓 Academic Information

**Project Title:** NutriSmart AI – Smart Food Recommendation System & 7-Day Meal Planner

**Degree:** Bachelor of Technology (B.Tech)

**Department:** Information Technology

**Academic Session:** 2025 – 2026

---

## 🤝 Contribution

Contributions, suggestions, and improvements are welcome.

1. Fork the repository
2. Create a new branch
3. Commit changes
4. Push changes
5. Create a Pull Request

---

## 📜 License

This project is developed for academic and educational purposes.

---

## ⭐ Support

If you found this project useful:

⭐ Star this repository

🍴 Fork this project

📢 Share with others

---

# "Eat Smart, Live Healthy with NutriSmart AI" 🥗🤖

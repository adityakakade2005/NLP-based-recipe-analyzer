🍳 Recipe Analyzer — An NLP Mini Project
A Natural Language Processing mini project that reads Indian recipes written as free text and turns them into structured information: ingredients with quantities, cooking actions, time, equipment, predicted regional cuisine, difficulty, a short summary and similar dishes.

Every lab experiment of the NLP course is one module of the final system, all in a single Jupyter notebook.

Author: Aditya Kakade — B.Tech (Artificial Intelligence & Data Science), Smt. Indira Gandhi College of Engineering, Navi Mumbai

Table of contents
Problem statement
Features
Experiments covered
Dataset
Project files
Installation
How to run
Using the analyzer
Results
Limitations
Future work
Credits
Problem statement
Recipes are unstructured text. Quantities, cooking actions, durations and equipment are buried inside sentences, which makes recipes hard to search, compare, categorise or summarise automatically.

Aim: build a Recipe Analyzer that takes raw recipe text and automatically

cleans and normalises it,
extracts ingredients (quantity, unit, name, Indian alias), cooking actions, durations, temperatures and equipment,
predicts the regional cuisine and the course (breakfast, dinner, snack…),
summarises the method and estimates difficulty,
analyses review sentiment (when a review is given),
finds and recommends similar dishes.
Features
Feature	How it works
Ingredient parsing	3 tablespoon Gram flour (besan) - sifted → quantity 3, unit tablespoon, name gram flour, alias besan
Recipe NER	Rules + an ingredient dictionary learned from the data (knows jeera, haldi, dhania, besan…)
Cooking actions	POS tagger + cooking-verb lexicon (soak → grind → ferment → serve)
Cuisine & course prediction	TF-IDF + Logistic Regression
Dish search	Find a dish by its Indian, English or Hindi name (rajma, masala dosa, बिरयानी) — tolerant of spelling variants
"What can I cook?"	Enter the ingredients you have (typos are corrected) and get matching dishes
Similar recipes	Cosine similarity on text + Jaccard similarity on ingredients
Summary & sentiment	Frequency-based extractive summary; VADER sentiment for reviews
Interfaces	Text menu and a point-and-click widget UI
Experiments covered
Exp	Topic	Role in the Recipe Analyzer
1	NLP applications & problem statement	Defines what the project does
2	Tokenization, filtration, script validation	Cleans input; detects Devanagari (Hindi) vs Latin text
3	Stop-word removal, stemming, lemmatization	Normalises recipe text
4	Morphological analysis & word generation	Cooking-verb and ingredient word forms
5	N-gram language model	Next-word suggestion, phrase statistics, perplexity
6	POS tagging (5 taggers compared)	Finds cooking actions
7	Chunking (features vs. training size)	Extracts ingredient / action phrases
8	Named Entity Recognition	Ingredients, durations, temperatures, equipment, methods
9	Text similarity	Dish lookup, recommendations, duplicate and typo detection
10	Word Sense Disambiguation with GRU / LSTM	pepper (spice vs vegetable), stock, date, roll, chip
★	Integration	RecipeAnalyzer class + interactive interfaces
Dataset
The notebook uses the public "6000+ Indian Food Recipes" dataset (recipes from Archana's Kitchen, compiled by Kanishka Jain; available on Kaggle and Hugging Face as nf-analyst/indian_recipe).

Dishes from every region of India with their Indian names, ingredients, step-by-step method, total time, course and diet.
After cleaning, 4,014 recipes across 41 regional cuisines are used. Non-Indian cuisines and rows that were never translated from Hindi are dropped.
Cleaning steps: fixing sentence spacing, splitting ingredient lists, separating the Indian dish name from the English description, keeping the Devanagari title in hindi_name.
The first run downloads the CSV automatically and caches it as IndianFoodDataset.csv. If you have no internet, download it manually (Kaggle: kanishk307/6000-indian-food-recipes-dataset) and save it with that name in the notebook folder.

Please check the dataset's licence/terms before redistributing it. It is used here for coursework.

Project files
.
├── Recipe_Analyzer_NLP_Mini_Project.ipynb   # the whole project (all 10 experiments + analyzer)
├── IndianFoodDataset.csv                    # created automatically on first run
└── README.md
Installation
Requirements: Python 3.10+ and Jupyter (Notebook, JupyterLab, VS Code or Google Colab).

pip install nltk pandas numpy matplotlib scikit-learn tensorflow ipywidgets jupyter
NLTK data (tokenizers, taggers, corpora) is downloaded by the first code cell of the notebook.

How to run
Open Recipe_Analyzer_NLP_Mini_Project.ipynb.
Run the cells from top to bottom (Run All). A full run takes roughly 3–5 minutes on a normal CPU; most of the time goes to pre-processing the corpus and training the GRU/LSTM models.
The last cells start the interactive analyzer.
On Google Colab everything except the dataset download works out of the box; upload the notebook and run all cells.

Using the analyzer
Text menu
 1) Analyze my own recipe
 2) Look up a dish by name (Indian / English / Hindi)
 3) What can I cook with the ingredients I have?
 4) Quit
Widget UI (optional last cell)
A tabbed interface built with ipywidgets:

✍️ Analyze my recipe — enter a title, ingredients (one per line), the method and an optional review.
🔎 Find a dish — search by Indian / English / Hindi name and open the full recipe with its analysis.
🥘 What can I cook? — list the ingredients you have and pick from the matching dishes.
If the widgets appear as plain text (VBox(children=...)), run pip install -U ipywidgets and restart the kernel. In VS Code, install the Jupyter extension.

Example output (abridged)
==================================================================
🍽  Paneer Butter Masala
==================================================================
Predicted    : region = North Indian (1.0) | course = Dinner (0.71)
Difficulty   : Easy (score 9.0)   Time: ~15.0 min mentioned in the method
Equipment    : kadhai
Actions      : heat → fry → chop → add → puree → cook → stir → simmer → garnish → serve
Ingredients (parsed):
  •    250 g            paneer [indian cottage cheese]
  •      1 cup          tomato puree
  •      2 tbsp         butter
  ...
Similar dishes: Tawa Paneer Masala (North Indian); Butter Chicken (North Indian); ...
Results
Task	Result
Regional cuisine classifier (held-out 20 %)	≈ 0.77 accuracy
Course classifier (held-out 20 %)	≈ 0.68 accuracy
POS taggers	Backoff n-gram taggers compared with the pre-trained averaged-perceptron tagger, which scored best
NP chunking	Context features (previous/next POS) give the biggest gain; more training data helps most at small sizes
Word sense disambiguation (GRU / LSTM)	100 % on the generated test split (see note below)
Exact numbers vary slightly between runs and environments.

Limitations
Class imbalance in region prediction: the 0.77 accuracy is lifted by the two large classes (North / South Indian). Smaller regions such as Tamil Nadu or Rajasthani have much lower recall, and regional cuisines overlap in reality (Punjabi ⊂ North Indian).
WSD data is generated from templates, so its accuracy is optimistic. It demonstrates the pipeline; use SemCor or a similar corpus for a real evaluation.
Rule-based NER works well on this corpus but will miss ingredients that are not in the learned dictionary.
Time estimate adds up every duration in the method (soaking and fermenting included), so it can be much larger than the listed total time.
Single-source dataset: the writing style is uniform. About 16 % of the Indian-cuisine rows were skipped because their text was never translated from Hindi.
The dataset has no reviews, so sentiment analysis only works on reviews typed in by the user.
Future work
Replace rule-based NER with a CRF / spaCy / transformer token classifier trained on annotated food data.
Add a Hindi pipeline (Devanagari tokenizer, Hindi WordNet) to use the untranslated recipes.
Unit conversion and calorie / nutrition estimation; dietary tags (vegan, gluten-free).
Serve the analyzer through Flask / FastAPI or Streamlit, or as a chatbot ("What can I cook with eggs and rice?").
Train the WSD models on a real sense-annotated corpus.
Credits
Recipe data: 6000+ Indian Food Recipes dataset (Archana's Kitchen; compiled by Kanishka Jain).
Libraries: NLTK, scikit-learn, pandas, NumPy, Matplotlib, TensorFlow/Keras, ipywidgets.
Corpora used in the experiments: Penn Treebank sample and CoNLL-2000 (via NLTK), WordNet, VADER lexicon.

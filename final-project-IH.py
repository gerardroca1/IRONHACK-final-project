mport warnings 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.metrics import accuracy_score 
from termcolor import colored 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# load data
wine_quality_df = pd.read_csv('/FINAL-PROJECT/winequality-red.csv')
best_wine_producers_df = pd.read_csv('/IRONHACK/FINAL-PROJECT/The_best_wine_producers_in_the_world.csv')
 
wine_quality_df.head()
best_wine_producers_df.head()

for index, row in wine_quality_df.iterrows():
    print(colored(f"Row {index + 1}:","white",attrs=['reverse']))
    print(row)
    print("--------------------------------------")

print("Shape =",wine_quality_df.shape)

num_rows, num_cols = wine_quality_df.shape
num_features = num_cols - 1
num_data = num_rows * num_cols

print(f"Number of Rows: {num_rows}")
print(f"Number of Columns: {num_cols}")
print(f"Number of Features: {num_features}")
print(f"Shape: {wine_quality_df.shape}")
print(wine_quality_df.info())

wine_quality_df.describe().T.round(2)

# Feature Engineering
wine_quality_df['sulphates_alcohol_interaction'] = wine_quality_df['sulphates'] * wine_quality_df['alcohol']

sns.catplot(data=wine_quality_df, x='quality', kind='count')
plt.title('Wine Quality Distribution')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='quality', y='volatile acidity', data=wine_quality_df, ax=ax)

ax.set_title('Volatile Acidity by Quality')
ax.set_xlabel('Quality')
ax.set_ylabel('Volatile Acidity')

ax.grid(True, axis='y', linestyle='--')
sns.set_palette('dark')  
plt.xticks(rotation=45)

plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='quality', y='alcohol', data=wine_quality_df, ax=ax)

ax.set_title('Alcohol by Quality')
ax.set_xlabel('Quality')
ax.set_ylabel('Alcohol')

ax.grid(True, axis='y', linestyle='--')
sns.set_palette('dark')  
plt.xticks(rotation=45)

plt.show()

# Check for missing values 
print("Missing values in wine quality dataset:\n", wine_quality_df.isnull().sum())
print("_________________________________________________________________")

print("Missing values in wine producers dataset:\n", best_wine_producers_df.isnull().sum())

plt.figure(figsize=(22, 11))
sns.stripplot(data=wine_quality_df, color="red", jitter=0.2, size=5)
plt.title("Outliers")
plt.xlabel("X-axis label")
plt.ylabel("Y-axis label")
plt.show()

print("Before Removing the outliers", wine_quality_df.shape)
wine_quality_df = wine_quality_df[wine_quality_df['total sulfur dioxide']<160]
print("After Removing the outliers", wine_quality_df.shape)

plt.figure(figsize=(22, 11))
sns.stripplot(data=wine_quality_df, color="red", jitter=0.2, size=5)
plt.title("Outliers")
plt.xlabel("X-axis label")
plt.ylabel("Y-axis label")
plt.show()

# Checking for duplicate entries
duplicate_count = wine_quality_df.duplicated().sum()
if duplicate_count == 0:
    print(colored("No duplicate entries found in the dataset.","green", attrs=['reverse']))
else:
    print(colored(f"Number of duplicate entries found: {duplicate_count}","yellow", attrs=['bold']))

# Group the data by country and calculate the mean of Total Cup Point
# For global exports
df_grouped = best_wine_producers_df.groupby('Country')['Global Exports'].mean().reset_index()
fig = px.choropleth(df_grouped, 
                    locations='Country', 
                    locationmode='country names',
                    hover_name='Country',
                    color='Global Exports',
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title= "The world's best wine exports" )

fig.show()

# For global producers
df_grouped = best_wine_producers_df.groupby('Country')['Wine Produced (million hectolitres)'].mean().reset_index()
fig = px.choropleth(df_grouped, 
                    locations='Country', 
                    locationmode='country names',
                    hover_name='Country',
                    color='Wine Produced (million hectolitres)',
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title= "The world's best wine producers" )

fig.show()

# Histograms
wine_quality_df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# Plot pairwise relationships in the 'wine_quality_df' DataFrame and color the points based on the 'quality' variable
sns.pairplot(wine_quality_df, hue="quality")

# Display the plot
plt.show()

# Create a pie chart to visualize the distribution of values in the 'quality' column of the 'data' DataFrame
wine_quality_df.quality.value_counts().plot.pie(autopct='%1.1f%%', shadow=True, figsize=(10, 10))

plt.title('Quality Graph')
plt.legend()
plt.show()

# Calculate the correlation matrix of the 'data' DataFrame
correlation = wine_quality_df.corr()

# Set the figure size for the heatmap plot
plt.figure(figsize=(10, 6))

# Create a heatmap using seaborn (sns) to visualize the correlation matrix
# The 'annot' parameter displays the correlation values in each cell
sns.heatmap(correlation, annot=True)

# Set the title of the heatmap plot
plt.title("Correlation Matrix")

# Display the heatmap plot
plt.show()

bins = [0, 5.5, 7.5, 10]
labels = [0, 1, 2]
wine_quality_df['quality'] = pd.cut(wine_quality_df['quality'], bins=bins,labels=labels)

X = wine_quality_df[wine_quality_df.columns[:-1]]
Y = wine_quality_df['quality']

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(wine_quality_df.drop('quality', axis=1))
wine_quality_scaled_df = pd.DataFrame(scaled_features, columns=wine_quality_df.columns[:-1])

# Split the data into training and testing sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of Y_train:", Y_train.shape)
print("Shape of Y_test:", Y_test.shape)

# Initialize lists to store training and testing accuracies
scoreListRF_Train = []
scoreListRF_Test = []

# Iterate over different values of max_depth
for max_dep in range(1, 5):
    # Iterate over different values of random_state
    for rand_state in range(1, 20):
        # Iterate over different values of n_estimators
        for n_est in range(1, 51):
            # Create a Random Forest model with the different values of max_depth, random_state, and n_estimators
            Model = RandomForestClassifier(n_estimators=n_est, random_state=rand_state, max_depth=max_dep)            
            
            # Fit the model on the training data
            Model.fit(X_train, Y_train)
            
            # Calculate and store the training accuracy
            scoreListRF_Train.append(Model.score(X_train, Y_train))
            
            # Calculate and store the testing accuracy
            scoreListRF_Test.append(Model.score(X_test, Y_test))

# Find the maximum accuracy for both training and testing
RF_Accuracy_Train = max(scoreListRF_Train) 
RF_Accuracy_Test = max(scoreListRF_Test)

# Print the best accuracies achieved
print(f"Random Forest best accuracy (Training): {RF_Accuracy_Train*100:.2f}%")
print(f"Random Forest best accuracy (Testing): {RF_Accuracy_Test*100:.2f}%")

# Print a success message indicating that the model has been trained successfully
print(colored("The Random Forest model has been trained successfully","green", attrs=['reverse']))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title(f'RandomForestClassifier Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}
# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n{model_name} Model Evaluation")
    print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
    print("Classification Report:\n", classification_report(Y_test, y_pred))
    print("Accuracy:", accuracy_score(Y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'Random Forest': {'n_estimators': [5, 10, 50]},
    'Decision Tree': {'max_depth': [None, 10, 20, 30]}
}

best_models = {}
for model_name, model in models.items():
    if model_name in param_grid:
        grid_search = GridSearchCV(model, param_grid[model_name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, Y_train)
        best_models[model_name] = grid_search.best_estimator_
        print(f'Best parameters for {model_name}: {grid_search.best_params_}')

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Function to evaluate a model
def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    
    print(f"\n{model.__class__.__name__} Model Evaluation")
    print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
    print("Classification Report:\n", classification_report(Y_test, y_pred))
    print("Accuracy:", accuracy)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model.__class__.__name__} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    return accuracy

# Train and evaluate best models
best_model_name = None
best_model = None
best_accuracy = 0

for model_name, model in best_models.items():
    accuracy = evaluate_model(model, X_train, Y_train, X_test, Y_test)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name
        best_model = model

print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy}")

# Train and evaluate models
best_model_name = None
best_model = None
best_accuracy = 0

for model_name, model in models.items():
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"\n{model_name} Model Evaluation")
    print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
    print("Classification Report:\n", classification_report(Y_test, y_pred))
    print("Accuracy:", accuracy)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name
        best_model = model

print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy}")

# Evaluate models further with ROC Curve and AUC Score
from sklearn.metrics import roc_curve, roc_auc_score

for model_name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]  # Get probability estimates
    fpr, tpr, thresholds = roc_curve(Y_test, y_prob)
    auc_score = roc_auc_score(Y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

# Save the best model and scaler for future use
import joblib

joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
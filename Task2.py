import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Define file path
file_path = 'D:\\Bhavesh goldi\\Python file\\train.csv'

# Check if file exists
if not os.path.exists(file_path):
    print(f"Error: The file {file_path} does not exist. Please download 'train.csv' from "
          f"https://www.kaggle.com/c/titanic/data and save it to {file_path}")
else:
    # Load the dataset
    df = pd.read_csv(file_path)

    # Data Cleaning
    print("Initial Data Info:")
    print(df.info())

    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)  # Impute missing Age with median
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Impute missing Embarked with mode
    df['Fare'].fillna(df['Fare'].median(), inplace=True)  # Impute missing Fare with median
    df.drop(columns=['Cabin', 'Ticket'], inplace=True)  # Drop Cabin and Ticket due to high missing data

    print("\nCleaned Data Info:")
    print(df.info())

    # Exploratory Data Analysis
    # 1. Survival rate by Passenger Class
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Pclass', y='Survived', data=df, ci=None, palette='viridis')
    plt.title('Survival Rate by Passenger Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Survival Rate')
    for i, v in enumerate(df.groupby('Pclass')['Survived'].mean()):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.show()

    # 2. Survival rate by Sex
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Sex', y='Survived', data=df, ci=None, palette='viridis')
    plt.title('Survival Rate by Sex')
    plt.xlabel('Sex')
    plt.ylabel('Survival Rate')
    for i, v in enumerate(df.groupby('Sex')['Survived'].mean()):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.show()

    # 3. Age distribution of survivors vs non-survivors
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=30, palette='viridis')
    plt.title('Age Distribution by Survival')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend(labels=['Did Not Survive', 'Survived'])
    plt.show()

    # 4. Fare distribution by Survival
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Survived', y='Fare', data=df, palette='viridis')
    plt.title('Fare Distribution by Survival')
    plt.xlabel('Survived (0 = No, 1 = Yes)')
    plt.ylabel('Fare')
    plt.yscale('log')  # Log scale for better visualization of fare outliers
    plt.show()

    # 5. Survival rate by Embarked
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Embarked', y='Survived', data=df, ci=None, palette='viridis')
    plt.title('Survival Rate by Embarked Port')
    plt.xlabel('Embarked Port')
    plt.ylabel('Survival Rate')
    for i, v in enumerate(df.groupby('Embarked')['Survived'].mean()):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.show()

    # 6. Correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of Numeric Variables')
    plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib



df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())
# print(df.describe())
print(df.info())
# print(df.tail())
# print(df.count())
# print(df.isna().sum())
# print(df.isnull().sum())



df['Age'] = df['Age'].fillna(df['Age'].median())
print(df['Age'])
print(df.isna().sum())

df['HasCabin'] = df['Cabin'].notna().astype(int)
df = df.drop(columns=['Cabin'])

# print(df.head())
# print(df.info())


df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# print(df.info())


df = df.drop(columns=['PassengerId'])
df = df.drop(columns=['Name'])
df = df.drop(columns=['Ticket'])

# print(df.head())


# encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
#
# encoded = encoder.fit_transform(df[['Embarked', 'Sex']])
# df = df.drop(columns=['Embarked', 'Sex'])
# df = pd.concat([df, encoded], axis=1) #made into a comment as we're making a pipeline.

df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['Family_Size'] == 1).astype(int)
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

sns.barplot(x="Sex", y="Survived", data=df)
plt.show()
sns.barplot(x="Pclass", y="Survived", data=df)
plt.show()
sns.violinplot(x="Survived", y="Fare", data=df)
plt.show()
sns.histplot(df, x="Age", hue="Survived", bins=30, kde=False, multiple="stack")
plt.show()

#
# print(df.head())
# print(df.isna().sum())

X = df.drop(columns=["Survived"])
y = df["Survived"]

enc_cols = ['Embarked', 'Sex']
num_cols = [c for c in X.columns if c not in enc_cols]

pre = ColumnTransformer(transformers=[('cat', OneHotEncoder(drop="first", handle_unknown='ignore'), enc_cols)
                                      ,("num", "passthrough", num_cols)])

pipe = Pipeline(steps=[('pre', pre),
                       ("clf", GradientBoostingClassifier(random_state=42))])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


param_grid = {
    "clf__n_estimators": [200, 300, 500],
    "clf__learning_rate": [0.03, 0.05 ,0.1],
    "clf__max_depth": [3, 5, 7],
    "clf__subsample": [0.8, 1.0]
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv = 5,
    scoring = "roc_auc",
    verbose = 1
)

grid.fit(X_train, y_train)
grid_best = grid.best_estimator_

y_pred = grid_best.predict(X_test)
y_proba = grid_best.predict_proba(X_test)

print("Best params:", grid.best_params_)
print("CV AUC:", grid.best_score_)
print("Test AUC:", roc_auc_score(y_test, grid_best.predict_proba(X_test)[:,1]))
print("Confusion matix results:" , confusion_matrix(y_test, y_pred))
print("Classification report:" , classification_report(y_test, y_pred, digits=3))
print("probability of the 1st person dying:", y_proba[0,0])
print("probability of the 10th person dying:", y_proba[9,0])
print("probability of the 100th person dying:", y_proba[99,0])

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
            xticklabels= ['Died', 'Survived'],
            yticklabels= ['Died', 'Survived'])
plt.ylabel("Actual"), plt.xlabel("Predicted")
plt.show()

importances = grid_best.named_steps['clf'].feature_importances_
feat_names = grid_best.named_steps['pre'].get_feature_names_out()
plt.figure(figsize=(12,10))
sns.barplot(x=feat_names, y=importances)
plt.tight_layout()
plt.title("Feature importance")
plt.show()

joblib.dump(grid_best, 'titanic_pipeline.joblib')









import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from backend import fetch_data, create_features

#Fetch Data
df = fetch_data()
query = "Saudi Aramco OR oil prices OR Middle East economy"
df, _ = create_features(df, query)

#Target
df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
df = df.dropna()

#Features
X = df[['return', 'ma_5', 'rsi', 'sentiment', 'volatility', 'momentum']]
y = df['target']

#Class Balance
ratio = (y == 0).sum() / (y == 1).sum()

#Models
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
xgb = XGBClassifier(
    eval_metric='logloss',
    scale_pos_weight=ratio
)

#train
rf.fit(X, y)
xgb.fit(X, y)

#Save
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(xgb, 'xgb_model.pkl')

print("Model trained and saved.")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import logging
import joblib
import optuna
import asyncio
from datetime import datetime
from lime import lime_tabular
from joblib import Parallel, delayed
from sklearn.inspection import permutation_importance
import boto3
from scipy.stats import ks_2samp
import os

# Setup logging
logger = logging.getLogger('MachineLearning')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class MachineLearningModel:
    def __init__(self):
        self.models = self._initialize_models()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.feature_selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
        self.model_version = datetime.now().strftime("%Y%m%d%H%M%S")
        self.explainer = None
        self.drift_threshold = 0.05  # Example threshold for drift detection
        self.previous_accuracy = 0.0

    def _initialize_models(self):
        rf = RandomForestClassifier()
        xgb = XGBClassifier()
        ridge = RidgeClassifier()
        svm = SVC(probability=True)
        base_models = [('rf', rf), ('xgb', xgb), ('ridge', ridge), ('svm', svm)]
        ensemble = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
        return ensemble

    async def fetch_and_preprocess_data(self, data_sources):
        # Fetch and preprocess data in parallel
        dfs = await asyncio.gather(*[asyncio.to_thread(pd.read_csv, source) for source in data_sources])
        processed_dfs = Parallel(n_jobs=-1)(delayed(self._clean_and_engineer_data)(df) for df in dfs)
        return pd.concat(processed_dfs)

    def _clean_and_engineer_data(self, df):
        df = self._clean_data(df)
        return self._feature_engineering(df)

    def _clean_data(self, df):
        # Implement data cleaning logic
        df.dropna(inplace=True)
        return df

    def _feature_engineering(self, df):
        df['price'] = df['close']  # use close prices explicitly

        df['vwap'] = (df['volume'] * df['price']).cumsum() / df['volume'].cumsum()
        df['macd'] = df['price'].ewm(span=12, adjust=False).mean() - df['price'].ewm(span=26, adjust=False).mean()
        df['rsi'] = 100 - (100 / (1 + df['price'].pct_change().rolling(window=14).mean()))
        df['bollinger'] = (df['price'] - df['price'].rolling(window=20).mean()) / df['price'].rolling(window=20).std()
        df['atr'] = df['price'].rolling(window=14).std()
        df['momentum'] = df['price'] - df['price'].shift(4)
        df['stochastic'] = (df['price'] - df['price'].rolling(window=14).min()) / (
            df['price'].rolling(window=14).max() - df['price'].rolling(window=14).min()
        )

        df.drop(columns=['price'], inplace=True)  # drop temp column if not needed elsewhere
        return df

    def train(self, df):
        X = df.drop(columns=['signal'])
        y = df['signal']

        # Feature Scaling and Selection pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('selector', RFE(estimator=RandomForestClassifier(), n_features_to_select=10)),
            ('model', self.models)
        ])

        # Hyperparameter tuning with TimeSeriesSplit (Optuna)
        def objective(trial):
            params = {
                'model__rf__n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
                'model__xgb__max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                'model__ridge__alpha': trial.suggest_loguniform('ridge_alpha', 0.01, 10.0),
                'model__svm__C': trial.suggest_loguniform('svm_C', 0.01, 10.0)
            }

            pipeline.set_params(**params)
            tscv = TimeSeriesSplit(n_splits=5)

            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_val)
                score = accuracy_score(y_val, preds)
                scores.append(score)

            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        # Set best parameters
        best_params = study.best_params
        pipeline.set_params(**{
            'model__rf__n_estimators': best_params['rf_n_estimators'],
            'model__xgb__max_depth': best_params['xgb_max_depth'],
            'model__ridge__alpha': best_params['ridge_alpha'],
            'model__svm__C': best_params['svm_C']
        })
        # Final fit on entire dataset
        pipeline.fit(X, y)
        self.models = pipeline

        # LIME explainer adjusted to pipeline output
        transformed_X = self.models[:-1].transform(X)
        self.explainer = lime_tabular.LimeTabularExplainer(
            transformed_X, mode='classification'
        )

        logger.info(f"✅ Best parameters: {best_params}")

    def predict(self, market_data):
        try:
            data = self._feature_engineering(pd.DataFrame([market_data]))
            
            pipeline_features = self.models.named_steps['selector'].feature_names_in_
            
            # Ensure all required features are present
            for col in pipeline_features:
                if col not in data.columns:
                    data[col] = 0
            data = data[pipeline_features]  # Ensure consistent column ordering

            transformed_data = self.models[:-1].transform(data)
            prediction = self.models.predict(transformed_data)
            confidence = self.models.predict_proba(transformed_data).max()

            try:
                explanation = self.explainer.explain_instance(
                    transformed_data[0], self.models.predict_proba
                )
                explanation_list = explanation.as_list()
                logger.info(f"Prediction explanation: {explanation_list}")
            except Exception as lime_err:
                logger.warning(f"LIME explanation failed: {lime_err}")
                explanation_list = []

            return prediction[0], confidence, explanation_list

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, 0.0, []

    def evaluate(self, df):
        X = df.drop(columns=['signal'])
        y = df['signal']

        transformed_X = self.models[:-1].transform(X)
        predictions = self.models.predict(transformed_X)

        accuracy = accuracy_score(y, predictions)
        precision, recall, fscore, _ = precision_recall_fscore_support(y, predictions, average='binary')
        rmse = mean_squared_error(y, predictions, squared=False)
        sharpe_ratio = (np.mean(predictions) - 0.0) / (np.std(predictions) + 1e-8)

        logger.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, RMSE: {rmse}, Sharpe Ratio: {sharpe_ratio}")

        # Statistical drift detection using KS test
        ks_stat, p_value = ks_2samp(y, predictions)
        if p_value < 0.05:
            logger.warning("⚠️ Significant drift detected (p-value < 0.05). Retraining recommended.")

        # update previous accuracy
        self.previous_accuracy = accuracy

    def save_model(self, path=None):
        if path is None:
            path = f'models/machine_learning_model_{self.model_version}.pkl'
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure directory exists
        joblib.dump(self.models, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path):
        self.models = joblib.load(path)
        logger.info(f"Model loaded from {path}")

    def adaptive_training(self, new_data):
        # Implement adaptive training logic
        self.train(new_data)

    def walk_forward_optimization(self, df, splits=5):
        tscv = TimeSeriesSplit(n_splits=splits)
        performance = []

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            logger.info(f"🚀 Walk-forward iteration {i+1}/{splits}")

            train, test = df.iloc[train_index], df.iloc[test_index]
            self.train(train)

            X_test = test.drop(columns=['signal'])
            y_test = test['signal']
            preds = self.models.predict(X_test)

            accuracy = accuracy_score(y_test, preds)
            logger.info(f"Iteration {i+1} Accuracy: {accuracy:.4f}")
            performance.append(accuracy)

        avg_perf = np.mean(performance)
        logger.info(f"✅ Average Walk-forward accuracy: {avg_perf:.4f}")

    def rolling_predictions(self, df):
        # Implement rolling predictions logic
        predictions = []
        for i in range(len(df) - 1):
            self.train(df.iloc[:i+1])
            prediction = self.predict(df.iloc[i+1])
            predictions.append(prediction)
        return predictions

    # def forecast_prices(self, historical_data):
    #     # Use the transformer model for forecasting
    #     forecast = self.time_series_model.predict(historical_data)
    #     return forecast
    # Check if the above code is actually needed or not and remove or keep accordingly

    def feature_importance_ranking(self, X, y):
        result = permutation_importance(self.models, X, y, n_repeats=10, random_state=42)
        sorted_idx = result.importances_mean.argsort()
        logger.info(f"Feature importances: {result.importances_mean[sorted_idx]}")
        return sorted_idx

    def monitor_performance(self, new_data):
        X_new = new_data.drop(columns=['signal'])
        y_new = new_data['signal']
        
        transformed_X = self.models[:-1].transform(X_new)
        predictions = self.models.predict(transformed_X)

        accuracy = accuracy_score(y_new, predictions)
        if abs(accuracy - self.previous_accuracy) > self.drift_threshold:
            logger.warning(f"⚠️ Performance drift detected: Previous accuracy={self.previous_accuracy:.4f}, Current accuracy={accuracy:.4f}")
            self.adaptive_training(new_data)
        
        self.previous_accuracy = accuracy

    def save_model_to_s3(self, bucket_name, model_name):
        s3 = boto3.client('s3')
        path = f'models/{model_name}.pkl'
        joblib.dump(self.models, path)
        s3.upload_file(path, bucket_name, model_name)
        logger.info(f"Model saved to S3 bucket {bucket_name} as {model_name}")

    def load_model_from_s3(self, bucket_name, model_name):
        s3 = boto3.client('s3')
        path = f'models/{model_name}.pkl'
        s3.download_file(bucket_name, model_name, path)
        self.models = joblib.load(path)
        logger.info(f"Model loaded from S3 bucket {bucket_name} as {model_name}")

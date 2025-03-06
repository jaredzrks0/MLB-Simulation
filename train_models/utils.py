from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


def ml_pipe(model=None):
        """
        Prepares the given dataset for machine learning by performing the following transformations:
        1. Defines the target variables (y_play and y_onbase) and drops them from the dataset.
        2. Scales numeric features and encodes categorical features (using one-hot encoding).
        3. Returns the preprocessed dataset along with the target variables.

        Args:
            final_dataset (pd.DataFrame): The dataset to be prepared for machine learning, which should
                                        include columns such as 'on_3b', 'on_2b', 'on_1b', 'inning_topbot',
                                        'game_date', 'play_type', and 'is_on_base'.

        Returns:
            dict: A dictionary containing:
                - "X" (np.ndarray): The preprocessed features (numeric and encoded categorical features).
                - "y_play" (pd.Series): The target variable for the type of play ('play_type').
                - "y_onbase" (pd.Series): The target variable indicating if the player is on base ('is_on_base').

        Example:
            prepared_data = make_dataset_machine_trainable(dataset)
            X = prepared_data["X"]
            y_play = prepared_data["y_play"]
            y_onbase = prepared_data["y_onbase"]
        """

        # Create a pipeline for scaling, encoding, and PCA

        numeric_transformer = Pipeline(
            steps=[("scaler", StandardScaler()),
                   ('dimensionality_reduction', PCA(n_components=.95))]
        )

        categorical_features = ["ballpark", "pitbat"]
        categorical_transformer = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Create a column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer,
                 make_column_selector(dtype_include="number")),
                ("cat", categorical_transformer,
                 make_column_selector(dtype_include="object")),
            ]
        )

        pipe = Pipeline(
            steps=[("preprocessor", preprocessor)]
        )

        if model != None:

            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

        return pipe
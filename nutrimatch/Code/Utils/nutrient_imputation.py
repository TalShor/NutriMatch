import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


def get_predictor(
        target:pd.DataFrame, train_subset:pd.Series, distances:pd.DataFrame, 
        **knn_kwargs):
    train_dist_matrix = distances.loc[train_subset, train_subset]
    train_target = target.loc[train_subset]
    
    # maybe use distances instead of uniform
    model = KNeighborsRegressor(metric='precomputed', **knn_kwargs)\
        .fit(train_dist_matrix, train_target)
    return model


def knn_impute_values(
        distances:pd.DataFrame, target:pd.DataFrame, train_subset:pd.Series,
        model:KNeighborsRegressor):

    target = target.copy()
    test_subset = target.index.difference(train_subset)
    test_dist_matrix = distances.loc[test_subset, train_subset]
    target.loc[test_subset] = model.predict(test_dist_matrix)
    return target


def impute_nutrients(nutrients:pd.DataFrame, distances:pd.DataFrame):
    all_imputed_values = []
    print(len(nutrients.columns))
    for i, nutrient_name in enumerate(nutrients.columns):
        print(i)
        nutrient = nutrients[[nutrient_name]] 
        nutrient_predictor = get_predictor(
            nutrient, nutrient.dropna().index, distances, 
            n_neighbors=1, weights='distance')
        imputed_values = knn_impute_values(
            distances, nutrient, nutrient.dropna().index, 
            model=nutrient_predictor)
        all_imputed_values.append(imputed_values)
    all_imputed_values = pd.concat(all_imputed_values, axis=1)
    return all_imputed_values

if __name__ == "__main__":
    full_nutrients_table = pd.read_parquet('s3://datasets-development/diet/food_registry/HPP/nutrients_from_zameret_fndds_sr_legacy/full_nutrients_table.parquet')
    distances = pd.read_parquet('s3://datasets-development/diet/food_registry/HPP/nutrients_from_zameret_fndds_sr_legacy/distances.parquet')
    distances = (1 - distances).clip(lower=0, upper=1)
    distances /= distances.max().max()
    distances_x2 = distances ** 2
    imputed_nutrients = impute_nutrients(full_nutrients_table, distances)
    imputed_nutrients.to_parquet('s3://datasets-development/diet/food_registry/HPP/nutrients_from_zameret_fndds_sr_legacy/imputed_nutrients_table.parquet')


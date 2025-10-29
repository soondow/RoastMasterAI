from imblearn.over_sampling import SMOTE

def get_smote(random_state: int=42, k_neighbors: int=5):
    return SMOTE(random_state=random_state, k_neighbors=k_neighbors)

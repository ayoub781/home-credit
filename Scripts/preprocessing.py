def preprocess_user_input(df):
    df['days_birth'] = abs(df['days_birth'])
    df['days_employed'] = df['days_employed'].replace(365243, 0)
    df['days_employed'] = abs(df['days_employed'])
    return df

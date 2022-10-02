try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import numpy as np
except Exception as e:
    print(e)

data = pd.read_csv("student_scores.csv")

# print(data)


y = data['Scores'].values.reshape(-1,1)
X = data['Hours'].values.reshape(-1, 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

r = LinearRegression()


r.fit(X_train, y_train)

# intercept = r.intercept_
# coefficient = r.coef_


def prediction(hrs):
    score = r.predict([[hrs]])

    return np.round(score, 1)[0][0]


print(prediction(9.5))











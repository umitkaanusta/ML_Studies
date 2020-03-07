import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# We will try to code the simple linear regression from scratch
# Linear equation = Theta0 + Theta1X1
# There are 1067 observations, we will train our model with first 800 and the remainder is the test set


def get_thetas():
    x = df_train["ENGINESIZE"].mean()
    y = df_train["CO2EMISSIONS"].mean()
    top = 0
    bottom = 0
    for i in range(len(df_train)):
        top += (df_train["ENGINESIZE"][i] - x) * (df_train["CO2EMISSIONS"][i] - y)
        bottom += (df_train["ENGINESIZE"][i] - x) ** 2
    theta_1 = top / bottom
    theta_0 = y - theta_1 * x
    thetas = (theta_0, theta_1)
    return thetas


def display_line(coef, intc):
    train_x = np.asanyarray(df_train[["ENGINESIZE"]])
    plt.scatter(df_train.ENGINESIZE, df_train.CO2EMISSIONS, color='blue')
    plt.plot(train_x, coef * train_x + intc, '-r')
    plt.xlabel("Engine size")
    plt.ylabel("CO2 Emission")
    plt.show()


def get_r_sq(coef, intc):
    pred_list = [df_test["ENGINESIZE"][i] * coef + intc for i in range(len(df_test["ENGINESIZE"]))]
    pred_arr = np.array(pred_list)
    rmse = np.sqrt(np.sum((pred_arr - df_test["CO2EMISSIONS"]) ** 2) / len(df_test["ENGINESIZE"]))
    r_squared = 100 - rmse
    return str(r_squared)


df = pd.read_csv("FuelConsumptionCo2.csv")
df = df[["ENGINESIZE", "CO2EMISSIONS"]]
df_train = df.iloc[range(0, 800)]
df_test = df.iloc[range(800, 1067)]
df_test = df_test.reset_index()
print(df.info())
print(df_train.info())
print(df_test.info())
intercept, coefficient = get_thetas()
print("y = {}x + {}".format(coefficient, intercept))
print("Accuracy of the model: " + get_r_sq(coefficient, intercept))
display_line(coefficient, intercept)

# y = 38.89375302126056x + 126.57977231890675
# Accuracy of the model: 72.97658424850084

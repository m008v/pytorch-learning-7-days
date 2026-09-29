import pandas as pd    
import numpy as np                                                                  # trực quan hóa dữ liệu
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
 

df = pd.read_csv('train.csv')
# _.columns =['Id','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','YearBuilt','FullBath','BedroomAbvGr','LotArea','SalePrice']
# print(_.head())
Id = df['Id']
df = df.drop('Id', axis=1)
X = df['SalePrice']
Y = df.drop(columns=['SalePrice'])

# Source - https://stackoverflow.com/a/18145399
# Posted by LondonRob, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-27, License - CC BY-SA 4.0




# sns.scatterplot(data=df, x='TotalBsmtSF', y='SalePrice', hue='OverallQual', palette='viridis')
# plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(Y, X, test_size=0.2, random_state=42)
print(X_train.head())
print(X_train.tail())

model = lr()
model.fit(X_train, Y_train)
test_pred = model.predict(X_test)
print('Predicted values:', test_pred)
print('Mean Squared Error:', mean_squared_error(Y_test, test_pred))
print('R2 Score:', r2_score(Y_test, test_pred))

public_test = pd.read_csv('public_test_input.csv')
print('Public test set shape:', public_test.head())
public_IdList = public_test['Id']
public_IdList_Drop = public_test.drop('Id', axis=1)
public_pred = model.predict(public_IdList_Drop)
print('Public test ID list:', public_IdList)
print('Predicted values for public test set:', public_pred)
ket_qua = pd.DataFrame({
    "id": public_IdList.to_numpy(),
    "price": np.round(public_pred).astype(int),
})
file_output = Path("results.csv")
ket_qua.to_csv(
    file_output,
    index=False,
    encoding="utf-8-sig",
)

print(f"Đã xuất {len(ket_qua)} dòng dữ liệu.")
print(f"File được lưu tại: {file_output.resolve()}")
print(ket_qua.head())
import pandas as pd

df = pd.read_csv("../datasets/Wine.csv")


def DropMissingValues():
    print("Amount of NaN values before execution\n", df.isnull().sum())
    df.dropna(how='any',
              inplace=True)  # Buradaki inplace kısmı gerçek veri üzerinde etkide bulunup bulunmama kararıdır.
    print("Amount of NaN values after execution\n", df.isnull().sum())
    return df  # Sutunlardaki toplam boş veri sayıları öğrenilir.


'''Eksik değer içeren verilerin medyan yardımı ile doldurulması'''


def MedianFillingFunc(self):
    df[self] = pd.to_numeric(df[self], errors='coerce')  # str verilerini NaN Formatına dönüştürür.
    median = df[self].median()
    df[self].fillna(median, inplace=True)
    print("Median of {} column is".format(self), median)
    return df


def ModeFillingFunc(self):
    mode = df[self].mode()
    df[self].fillna(mode, inplace=True)
    print("Mode of {} column is".format(self), mode)
    return df


DropMissingValues()

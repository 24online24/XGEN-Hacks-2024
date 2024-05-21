import polars as pl

df1 = pl.read_csv("ML/csv_train/Fake.csv", separator=',')
df2 = pl.read_csv("ML/csv_train/True.csv", separator=',')

df1 = df1.with_columns(
    pl.lit(0).alias("Label")
)

df2 = df2.with_columns(
    pl.lit(1).alias('Label')
)

combine_df = pl.concat([df1, df2]).drop(["date", "subject"])

combine_df.write_csv('ML/csv_train/Combined.csv')
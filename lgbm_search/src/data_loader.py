import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from IPython import embed
import math

class DataLoader:
    def __init__(self, path):
        self.path = path
        self.data = None
        self.series_dict = None
        self.scaler = Scaler()

    def load_data(self):

        table = pq.read_table(self.path)
        new_fields = []
        new_columns = []

        for i, field in enumerate(table.schema):
            col = table.column(i)
            # Manejo para tipos 'dbdate'
            if hasattr(field.type, "extension_name") and field.type.extension_name == "dbdate":
                cast_type = field.type.storage_type
                col = col.cast(cast_type)
                new_field = pa.field(field.name, cast_type)
            else:
                new_field = field

            new_fields.append(new_field)
            new_columns.append(col)

        new_schema = pa.schema(new_fields)
        new_table = pa.Table.from_arrays(new_columns, schema=new_schema)

        self.data = new_table.to_pandas()
        return self.data

    def to_week(self, limit=100):

        self.data['KeyDate'] = pd.to_datetime(self.data['KeyDate'])
        # Ajustar KeyDate al inicio de la semana (lunes)
        self.data['KeyDate'] = self.data['KeyDate'] - pd.to_timedelta(self.data['KeyDate'].dt.dayofweek, unit='d')

        # Filtramos solo historia
        self.data = self.data[self.data['DateFlag'] == 'History']

        columns_to_drop = [
            'KeyCurrency',
            'HolidayName',
            'HolidayOrPromoIndex',
            'DayNumberInMonth',
            'DayNumberInWeek',
            'Processor',
            'KeyProduct']

        self.data = self.data.drop(columns=columns_to_drop)

        group_cols = [
            'KeySupplyChain',
            'TSType',
            'Super_Seasonal',
            'IntermittenceRatio',
            'DaysWithSales',
            'Length',
        ]

        grouped_df = (
            self.data
            .groupby(group_cols + ['KeyDate'], as_index=False)
            .agg(
                Demand_Sum=('Demand', 'sum'),
                CPI_Mean=('CPI', 'mean'),
                Price_Mean=('Price', 'mean')
            )
        )

        grouped_df = grouped_df.sort_values('KeyDate')

        self.series_dict = {}
        count_series = 0

        for keys, df_sub in grouped_df.groupby(group_cols):
            df_sub = df_sub.sort_values('KeyDate')

            try:
                ts = TimeSeries.from_dataframe(
                    df_sub,
                    time_col='KeyDate',
                    value_cols=['Demand_Sum'],
                    freq='W-MON'
                )
                self.series_dict[keys] = ts
            except ValueError as e:
                print(f"Error creando TimeSeries para la llave {keys}: {e}")

            count_series += 1
            if limit and count_series >= limit:
                break

        print(f"Se crearon {len(self.series_dict)} series de tiempo.")

    def prepare_data_for_forecasting(self, scaling=False):
        train_series = {}
        val_series = {}
        test_series = {}

        for keys, series in self.series_dict.items():
            total_points = len(series)
            if total_points < 5:
                print(f"Skipping {keys} only has {total_points} points.")
                continue

            test_size = math.ceil(total_points * 0.05)
            val_size = math.ceil(total_points * 0.05)
            test_size = max(test_size, 1)
            val_size = max(val_size, 1)

            te_ts = series[-test_size:]
            v_ts  = series[-(test_size + val_size):-test_size]
            tr_ts = series[:-(test_size + val_size)]

            if len(tr_ts) > 0:
                train_series[keys] = tr_ts
            if len(v_ts) > 0:
                val_series[keys] = v_ts
            if len(te_ts) > 0:
                test_series[keys] = te_ts

        return train_series, val_series, test_series

from hour_type import HourType

hour_types = {"nyc_taxi": HourType.TO_MINUTE,
              "power": HourType.TO_MINUTE,
              "energy": HourType.TO_MINUTE,
              "reddit": HourType.TO_MINUTE,
              "retail": HourType.TO_HOUR}

skip_non_floats = ["power"]

predicted_fields = {"nyc_taxi": "passenger_count",
                    "sunspot": "spots",
                    "dodger": "count",
                    "power": "Global_active_power",
                    "energy": "T3",
                    "retail": "turnover",
                    "reddit": "count",
                    "test": "y"}

date_formats = {"nyc_taxi": '%Y-%m-%d',
                "sunspot": '%Y-%m-%d %H:%M:%S',
                "dodger": '%Y-%m-%d %H:%M:%S',
                "power": '%d/%m/%Y',
                "energy": '%Y-%m-%d',
                "retail": '%d/%m/%Y',
                "reddit": '%Y-%m-%d',
                "test": "%Y-%m-%d"
                }

data_skips = {"nyc_taxi": [1,2],
              "sunspot": [1,],
              "dodger": [1,2],
              "power": [1,],
              "energy": [1,],
              "retail": [1,],
              "reddit": [1,],
              "test": []
              }


error_ignore_first = {"nyc_taxi": 3000,
                    "sunspot": 5000,
                    "dodger": 5000,
                    "power": 5000,
                    "energy": 5000,
                    "retail": 1000,
                    "reddit": 5000,
                    "test": 5000}
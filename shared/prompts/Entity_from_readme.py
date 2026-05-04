README_EXTRACT_JSON_PROMPT = """
-Goal-
Identify and categorize all entities given the dataset readme file.


Guidelines:
1. Each entity should be mapped to exactly one of these categories: {entity_types}
2. Provide clear, concise descriptions that capture the measurement purpose
3. Ensure entity_id matches the exact name from the source
4. Generate valid JSON output



Required JSON format for each entity:
{
    "name": "brief descriptive name",
    "id": "exact_source_name",
    "type": "category from provided types",
    "description": "clear explanation of what is measured",
    "range": "range of the measurement",
    "unit": "unit of the measurement",
    "path": "path to where the data is stored, exactly what's after ###",
    "source": "source of the data",
    "time_or_date": "Represents the type of temporal data recorded in the column.timestamp: Indicates the data was collected at a specific date and time.date: Represents per-day aggregated data without time details"
}

######################
-Examples-
######################
Example 1:
Entity_types: Sleep, Physical Activity, Behavioral, Physiological
Text:
### oura/heart\_rate.csv


| **Column name** |                                                  **Description**                                                  | **Range** |     **Type (unit)**     |
| :--------------: | :---------------------------------------------------------------------------------------------------------------: | :-------: | :----------------------: |
|  **timestamp**  |                                         Epoch timestamp of this data row.                                         |     -     | Timestamp (milliseconds) |
| **heart\_rate** |                            Average heart rate for each 5 minutes of the sleep period.                            |     -     |    Number (beats/min)    |
| **heart\_rmssd** | The average root mean square of successive heartbeat interval differences for each 5 minutes of the sleep period. |     -     |  Number (milliseconds)  |######################
Output:
[
{
    "name": "sleep heart rate",
    "id": "heart_rate", 
    "type": "Physiological",
    "description": "Average heart rate measured during sleep periods in beats per minute",
    "range": "None",
    "unit": "beats/min",
    "path": "oura/heart_rate.csv",
    "source": "oura",
    "timestamp": "timestamp"
}
{
    "name": "sleep successive heartbeat interval differences",
    "id": "heart_rmssd",
    "type": "Physiological",
    "description": "The average root mean square of successive heartbeat interval differences for each 5 minutes of the sleep period.",
    "range": "None",
    "unit": "milliseconds",
    "path": "oura/heart_rate.csv",
    "source": "oura",
    "timestamp": "timestamp"
}
]

-Real Data-
######################
"""


FEATURE_RELATIONSHIPS_GENERATION_PROMPT = """
-Goal-
Identify all features. For each identified feature, extract the following information:
- feature_name: Name of the feature
- feature_type: One of the following types: [{feature_types}]
- feature_description: description of the feature
Format each feature as ("feature"{{tuple_delimiter}}<feature_name>{{tuple_delimiter}}<feature_type>{{tuple_delimiter}}<feature_description>)

######################
-Examples-
######################
Example 1:
Feature_types: SLEEP, ACTIVITY, READINESS, PHYSIOLOGICAL
Text:
### oura/heart\_rate.csv


| **Column name** |                                                  **Description**                                                  | **Range** |     **Type (unit)**     |
| :--------------: | :---------------------------------------------------------------------------------------------------------------: | :-------: | :----------------------: |
|  **timestamp**  |                                         Epoch timestamp of this data row.                                         |     -     | Timestamp (milliseconds) |
| **heart\_rate** |                            Average heart rate for each 5 minutes of the sleep period.                            |     -     |    Number (beats/min)    |
| **heart\_rmssd** | The average root mean square of successive heartbeat interval differences for each 5 minutes of the sleep period. |     -     |  Number (milliseconds)  |######################
Output:
("feature"{{tuple_delimiter}}heart_rate{{tuple_delimiter}}PHYSIOLOGICAL{{tuple_delimiter}}Average heart rate for each 5 minutes of the sleep period.-unit in beats/min)
{{record_delimiter}}
("feature"{{tuple_delimiter}}heart_rmssd{{tuple_delimiter}}PHYSIOLOGICAL{{tuple_delimiter}}The average root mean square of successive heartbeat interval differences for each 5 minutes of the sleep period. Unit in milliseconds.)
{{completion_delimiter}}

-Real Data-
######################
feature_types: {feature_types}
text: {input_text}
######################
output:
"""

FEATURE_GENERATION_JSON_PROMPT = """
-Goal-
Identify all features given the dataset readme file. For each identified feature, extract the following information:
- feature_name: feature name summarizing what the feature measures 
- feature_id: exact feature name from text
- feature_type: One of the following types: [{feature_types}]
- feature_description: description of the feature

Format each entity output as a JSON entry with the following format:

{{"name": <feature_name>, "id": <feature_id>, "type": <feature_type>, "description": <feature_description>}}

######################
-Examples-
######################
Example 1:
Feature_types: SLEEP, ACTIVITY, READINESS, PHYSIOLOGICAL
Text:
### oura/heart\_rate.csv


| **Column name** |                                                  **Description**                                                  | **Range** |     **Type (unit)**     |
| :--------------: | :---------------------------------------------------------------------------------------------------------------: | :-------: | :----------------------: |
|  **timestamp**  |                                         Epoch timestamp of this data row.                                         |     -     | Timestamp (milliseconds) |
| **heart\_rate** |                            Average heart rate for each 5 minutes of the sleep period.                            |     -     |    Number (beats/min)    |
| **heart\_rmssd** | The average root mean square of successive heartbeat interval differences for each 5 minutes of the sleep period. |     -     |  Number (milliseconds)  |######################
Output:
[
  {{"name": "sleep heart rate", "id": "heart_rate", "type": "PHYSIOLOGICAL", "description": "Average heart rate for each 5 minutes of the sleep period.-unit in beats/min"}},
  {{"name": "sleep successive heartbeat interval differences", "id": "heart_rmssd", "type": "PHYSIOLOGICAL", "description": "The average root mean square of successive heartbeat interval differences for each 5 minutes of the sleep period. Unit in milliseconds."}}
]

-Real Data-
######################
feature_types: {feature_types}
text: {input_text}
######################
output:
"""


FEATURE_GENERATION_JSON_PROMPT2 = """
-Goal-
Identify all features given the dataset readme file. For each identified feature, extract the following information:
- feature_name: feature name summarizing what the feature measures 
- feature_id: exact feature name from text
- feature_type: One of the following types: [{feature_types}]
- feature_description: description of the feature
- data_path: path to where the data is stored, exactly what's after ###

Format each entity output as a JSON entry with the following format:

{{"name": <feature_name>, "id": <feature_id>, "type": <feature_type>, "description": <feature_description>},"path":<data_path>}}

######################
-Examples-
######################
Example 1:
Feature_types: SLEEP, ACTIVITY, READINESS, PHYSIOLOGICAL
Text:
### oura/heart\_rate.csv


| **Column name** |                                                  **Description**                                                  | **Range** |     **Type (unit)**     |
| :--------------: | :---------------------------------------------------------------------------------------------------------------: | :-------: | :----------------------: |
|  **timestamp**  |                                         Epoch timestamp of this data row.                                         |     -     | Timestamp (milliseconds) |
| **heart\_rate** |                            Average heart rate for each 5 minutes of the sleep period.                            |     -     |    Number (beats/min)    |
| **heart\_rmssd** | The average root mean square of successive heartbeat interval differences for each 5 minutes of the sleep period. |     -     |  Number (milliseconds)  |######################
Output:
[
  {{"name": "sleep heart rate", "id": "heart_rate", "type": "PHYSIOLOGICAL", "description": "Average heart rate for each 5 minutes of the sleep period.-unit in beats/min", "path":oura/heart_rate.csv}},
  {{"name": "sleep successive heartbeat interval differences", "id": "heart_rmssd", "type": "PHYSIOLOGICAL", "description": "The average root mean square of successive heartbeat interval differences for each 5 minutes of the sleep period. Unit in milliseconds.", "path":oura/heart_rate.csv}}
]

-Real Data-
######################
feature_types: {feature_types}
text: {input_text}
######################
output:
"""
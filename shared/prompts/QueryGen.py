
# for each metric, generate a list of objective questions that can be answered.
QUERY_GENERATION_JSON_PROMPT = """
-Goal-
Given a wearable health metric, generate a list of questions that can be answered.
falls into the following categories:

General Knowledge
examples:
"what is the optimal range for HRV?"
"What is HRV?"

Personal data:
examples:
"What was my step count this week/yesterday/month?"
"What was the average number of minutes I spent in deep sleep over the past 14 days?"
"What was my percentage of light sleep today?"
"What was my max HR this week?"

Comparative Analysis
examples:
"How has my readiness score changed this week compared to last week?"
"Compare my deep sleep duration last week with this week."

Pattern Detection
examples:
"Identify any trends in my daily movement over the last 30 days."

Anomaly Detection
examples:
"Did anything unusual happen in my sleep metrics this month?"
"Are there any spikes or drops in my activity levels that might need attention?"
"Was there an abnormal recovery time recorded this week?"

-Input Format- 
{
    "name": "<entity_name>", 
    "description": "<entity_description>"
}

-Output Format-
[
    {
        "question": "<question>",
        "answer": "<answer>",
        "objective": <boolean>,
        "openness": <openness_score>,
        "related_nodes": ["<related_node_1>", "<related_node_2>", "<related_node_3>"]
    },
    ...
]

"""

QUERY_GENERATION_JSON_PROMPT_SINGLE_ENTITY = """
Generate diverse, clinically relevant questions about health metrics from wearable data. 

INPUT FORMAT (Array of metric objects):
[
    {
        "id": "<unique_id>",
        "name": "<metric_name>",        # The health metric being analyzed
        "description": "<definition>",  # Clinical definition of the metric
        "date": "<YYYY-MM-DD>",
        "time_granularity": "<1|7|14|30|60|all>",  # Time period covered
        "abnormality_level": "<low|medium|high>", # Deviation from user's baseline
    },
    ...
]


OUTPUT FORMAT (Array of questions - one per input metric):
[
    {
        "id": "<matching_input_id>",
        "question": "<clear, time-bound phrasing>",
        "question_type": "<one of: General Knowledge | Data Retrieval | Trend Analysis | Comparative Insight | Anomaly Detection | Actionable Advice | Exploratory Analysis>",
        "openness": <0.0-1.0>,     # 0.0=closed, 1.0=open-ended
    },
    ...
]
QUESTION FRAMEWORK:
1. **General Knowledge** (Openness: 0.2-0.4)
   - Definitions, benchmarks, normal ranges
   - Example: "What's considered a healthy range for [metric]?" "What is [metric]?"

2. **Data Retrieval** (Openness: 0.1-0.3)
   - Specific time-bound numerical queries
   - Example: "What was my [metric] yesterday?" "What was my max/min/average [metric] this week?"

3. **Trend Analysis** (Openness: 0.4-0.6)
   - Patterns over days/weeks/months
   - Example: "Identify any trends in my [metric] over the last 30 days." "Summarize my [metric] for the past month."

4. **Comparative Insight** (Openness: 0.5-0.7)
   - Time-period comparisons
   - Example: "How does this week's [metric] compare to last week?"

5. **Anomaly Detection** (Openness: 0.6-0.8)
   - Statistical outliers
   - Example: "Were there unusual [metric] spikes in this month?"

6. **Actionable Advice** (Openness: 0.3-0.5)
   - Data-driven recommendations
   - Example: "What adjustments could improve my [metric]?"

7. **Exploratory Analysis** (Openness: 0.7-1.0)
   - Multi-factor investigations
   - Example: "Do you think I am stressed recently?" "I'm feeling really tired today. do you know why?" "Why might I be feeling tired despite sleeping 8 hours?"


GENERATION RULES:
1. Time binding:
   - Map granularity to natural terms:
     • 1 → "today"
     • 7 → "past 7 days"
     • 14 → "past 14 days"
     • 30 → "past 30 days"
     • 60 → "past 60 days"
     • all → "overall"
2. Blend concrete and exploratory questions per category:
   - 40% objective (openness ≤0.4)
   - 30% moderate (0.4 < openness <0.7)
   - 30% open-ended (≥0.7)
3. Prevent overlap between categories
4. For medium/high abnormalities, prioritize generating high openness questions
5. Exactly 1 output question per input group

EXAMPLES:

INPUT:
[
        {
            "id": "m001",
            "name": "Inactive time",  
            "description": "The amount of time a user is inactive, measured in minutes",
            "date": "2020-01-01",
            "time_granularity": "1",
            "abnormality_level": "low", 
        },
        {
            "id": "m002",
            "name": "total sleep time",  
            "description": "The total amount of time a user spends in sleep",
            "date": "2020-02-02",
            "time_granularity": "14",
            "abnormality_level": "high", 
        },
        ...
    ]
]
OUTPUT:
[
    {
        "id": "m001",
        "question": "What was my inactive time today?",
        "question_type": "Data Retrieval",
        "openness": 0.1,
    },
    {
        "id": "m002",
        "question": "How does my total sleep time over the past 14 days compare to the previous period?",
        "question_type": "Comparative Insight",
        "openness": 0.7,
    },
    ...
]

"""



QUERY_GENERATION_JSON_PROMPT_MULTIPLE_ENTITIES = """
Generate clinically relevant questions from wearable data, with each question containing 2-3 metrics.

INPUT FORMAT (Array of metric objects):
[
    {
        "id": "<unique_id>",
        "metrics":[
            {
                "name": "<metric_name_1>",        # The health metric being analyzed
                "description": "<definition>",  # Clinical definition of the metric
            },
            {
                "name": "<metric_name_2>",        # The health metric being analyzed
                "description": "<definition>",  # Clinical definition of the metric
            },
            ...
        ],
        "date": "<YYYY-MM-DD>",
        "time_granularity": "<1|7|14|30|60|all>",  # Time period covered
    },
    ...
]


OUTPUT FORMAT (Array of questions - one per input):
[
    {
        "id": "<matching_input_id>",
        "question": "<clear, time-bound phrasing>",
        "question_type": "<one of: Metric Relationships | Contextual Queries>",
        "openness": <0.0-1.0>,     # 0.0=closed, 1.0=open-ended
    },
    ...
]
QUESTION FRAMEWORK:
1. **Metric Relationships** (Openness: 0.4-0.6)   
   - Example: "Does [metric1] relate to [metric2] trends for the past 30 days?"  

2. **Contextual Queries** (Openness: 0.5-0.7)    
   - Example: "Do [metric1] spikes follow days with high [metric2]?" "Is there a pattern in my [metric1] on days I have a higher [metric2] for the past week?"


GENERATION RULES:
1. Time binding:
   - Map granularity to natural terms:
     • 1 → "today"
     • 7 → "past 7 days"
     • 14 → "past 14 days"
     • 30 → "past 30 days"
     • 60 → "past 60 days"
     • all → "overall"
2. Each question must reference metrics from the input list
3. Exactly 1 output question per input group




EXAMPLES:

INPUT:
[
    {
        "id": "001",
        "metrics": [
            {
                "name": "resting_heart_rate",
                "description": "Beats per minute at complete rest"
            },
            {
                "name": "sleep_duration", 
                "description": "Total minutes of sleep per night"
            }
        ],
        "date": "2023-11-15",
        "time_granularity": "30"
    }
]

OUTPUT:
[
    {
        "id": "001",
        "question": "How does my resting heart rate vary with sleep duration for the past 30 days?",
        "question_type": "Metric Relationships",
        "openness": 0.5,
    }
]

"""





# for multiple queries on a single entity
QUERY_GENERATION_JSON_PROMPT_SINGLE_ENTITY_V2 = """
TASK: Generate diverse, answerable questions about a health metric from wearable data.

INPUT FORMAT:
{
    "name": "<metric_name>",        # The health metric being analyzed
    "description": "<definition>",  # Clinical definition of the metric
    "data_points": [
        {
            "id": "<unique_id>",
            "date": "<YYYY-MM-DD>",
            "time_granularity": "<1|7|14|30|60|all>",  # 1:day, 7:week, 14:2weeks, 30:month, 60:2months, all:all time
            "abnormality_level": "<low|medium|high>"
        },
        ...
    ]
}

OUTPUT FORMAT:
[
    {
        "id": "<unique_id>",
        "question": "<clear, time-bound phrasing>",
        "question_type": "<one of: General Knowledge | Data Retrieval | Trend Analysis | Comparative Insight | Anomaly Detection | Actionable Advice | Exploratory Analysis>",
        "openness": <0.0-1.0>,     # 0.0=closed, 1.0=open-ended
    },
    ...
]
QUESTION FRAMEWORK:
1. **General Knowledge** (Openness: 0.2-0.4)
   - Definitions, benchmarks, normal ranges
   - Example: "What's considered a healthy range for [metric]?" "What is [metric]?"

2. **Data Retrieval** (Openness: 0.1-0.3)
   - Specific time-bound numerical queries
   - Example: "What was my [metric] yesterday?" "What was my max/min/average [metric] this week?"

3. **Trend Analysis** (Openness: 0.4-0.6)
   - Patterns over days/weeks/months
   - Example: "Identify any trends in my [metric] over the last 30 days." "Summarize my [metric] for the past month."

4. **Comparative Insight** (Openness: 0.5-0.7)
   - Time-period comparisons
   - Example: "How does this week's [metric] compare to last week?"

5. **Anomaly Detection** (Openness: 0.6-0.8)
   - Statistical outliers
   - Example: "Were there unusual [metric] spikes in this month?"

6. **Actionable Advice** (Openness: 0.3-0.5)
   - Data-driven recommendations
   - Example: "What adjustments could improve my [metric]?"

7. **Exploratory Analysis** (Openness: 0.7-1.0)
   - Multi-factor investigations
   - Example: "Do you think I am stressed recently?" "I'm feeling really tired today. do you know why?" "Why might I be feeling tired despite sleeping 8 hours?"


GENERATION RULES:
1. Always include time_range (1/7/14/30/60/all)
2. Blend concrete and exploratory questions per category:
   - 40% objective (openness ≤0.4)
   - 30% moderate (0.4 < openness <0.7)
   - 30% open-ended (≥0.7)
3. Prevent overlap between categories

EXAMPLES:

INPUT:
{
    "name": "Inactive time",  
    "description": "The amount of time a user is inactive, measured in minutes",
    "data_points": [
        {
            "id": "1",
            "date": "2020-01-01",
            "time_granularity": "1",
            "abnormality_level": "low", 
        },
        {
            "id": "2",
            "date": "2020-02-02",
            "time_granularity": "14",
            "abnormality_level": "high", 
        },
        ...
    ]
}
OUTPUT:
[
    {
        "id": "1",
        "question": "What was my inactive time yesterday?",
        "question_type": "Data Retrieval",
        "openness": 0.1,
    },
    {
        "id": "2",
        "question": "How does my inactive time over the past two weeks compare to the previous period?",
        "question_type": "Comparative Insight",
        "openness": 0.7,
    },
    ...
]

"""


# comparing all existing nodes at once 
ENTITY_MERGE_JSON_PROMPT_ALL = """
-Goal-
Analyze a new node against existing nodes to identify potential duplicates in a wearable health knowledge graph.

GUIDELINES FOR COMPARISON:
1. Semantic Analysis:
   - Look beyond exact text matches
   - Consider medical synonyms and related terms
   - Evaluate contextual meaning in healthcare

2. Description Analysis:
   - Identify overlapping concepts
   - Consider complementary information
   - Evaluate scope and specificity

4. Scoring Criteria:
   0.0-0.3: Clearly different concepts
   0.4-0.6: Related but distinct
   0.7-0.8: Highly similar
   0.9-1.0: Virtually identical

INPUT FORMAT:
{
    "input_name": "new node name",
    "input_description": "new node description",
    "references": [
        {
            "name": "existing node 1 name",
            "description": "existing node 1 description"
        },
        {
            "name": "existing node 2 name",
            "description": "existing node 2 description"
        }
        ...
    ]
}

OUTPUT FORMAT:
[
    {
        "input_name": "new node name",
        "reference_name": "matched existing node name",
        "similarity_score": <float 0-1>,
        "same_concept": <boolean>,
        "reasoning": "clear explanation of similarity assessment and why the nodes are the same or different"
    }
    ...
]

EXAMPLE :
Input:
{
    "input_name": "steps",
    "input_description": "Total number of steps registered during the day.",
    "references": [
    {
      "name": "Steps taken",
      "description": "Steps Taken refers to the total number of steps registered by a wearable device over a given period, typically a day. It is a key metric for assessing physical activity levels, with higher step counts generally associated with better cardiovascular health, weight management, and overall fitness. Wearable devices track steps using accelerometers or gyroscopes to detect motion and count steps based on movement patterns."
    },
    {
      "name": "Distance traveled",
      "description": "Distance Traveled refers to the total length of the path covered by an object or individual as it moves from one point to another over a given period of time. It is a scalar quantity, meaning it only considers the magnitude of the movement and not the direction. This metric is commonly measured by wearable devices using accelerometers and GPS to track movement and calculate the cumulative distance covered during activities such as walking, running, or cycling."
    },
    {
      "name": "Energy expenditure",
      "description": "Energy Expenditure (EE) refers to the total amount of energy, measured in kilocalories (kcal), that an individual expends over a 24-hour period. It comprises three main components: Basal Metabolic Rate (BMR), the energy required for basic physiological functions at rest; the Thermic Effect of Food (TEF), the energy used to digest and metabolize food; and Physical Activity-Related Energy Expenditure (PAEE), the energy expended during physical activities. EE is a critical metric for understanding caloric balance, weight management, and overall metabolic health. Wearable devices estimate EE using accelerometers, heart rate monitors, and algorithms that combine movement data with physiological parameters."
    },
    {
      "name": "Activity level",
      "description": "Activity Level (AL) quantifies the intensity of physical activity based on Metabolic Equivalent of Task (MET) levels, ranging from sedentary (non-wear) to high-intensity activity. It is a key metric for assessing daily physical activity patterns, energy expenditure, and overall fitness. Wearable devices estimate AL using accelerometers and algorithms that classify movement intensity into categories such as sedentary, lightly active, moderately active, and very active."
    }
  ]
}
Output:
[
    {
      "input_name": "steps",
      "reference_name": "Steps taken",
      "similarity_score": 0.95,
      "same_concept": true,
      "reasoning": "Both nodes refer to the total number of steps registered by a wearable device over a given period, typically a day. The descriptions are nearly identical, with both emphasizing the use of accelerometers or gyroscopes to detect motion and count steps. The terms 'steps' and 'steps taken' are semantically equivalent in this context."
    },
    {
      "input_name": "steps",
      "reference_name": "Distance traveled",
      "similarity_score": 0.4,
      "same_concept": false,
      "reasoning": "While both nodes are related to physical activity, 'steps' specifically refers to the number of steps taken, whereas 'distance traveled' refers to the total length of the path covered. These are distinct metrics, with 'distance traveled' often derived from steps but also incorporating additional data like stride length or GPS information."
    },
    {
      "input_name": "steps",
      "reference_name": "Energy expenditure",
      "similarity_score": 0.3,
      "same_concept": false,
      "reasoning": "The nodes are related to physical activity but measure different aspects. 'Steps' quantifies the number of steps taken, while 'energy expenditure' measures the total energy used, often derived from a combination of steps, heart rate, and other factors. These are complementary but distinct metrics."
    },
    {
      "input_name": "steps",
      "reference_name": "Activity level",
      "similarity_score": 0.5,
      "same_concept": false,
      "reasoning": "Both nodes are related to physical activity, but 'steps' is a specific count of steps taken, while 'activity level' quantifies the intensity of physical activity based on MET levels. They are related but distinct, as 'activity level' may incorporate steps but also considers other factors like movement intensity."
    }
]

"""


ENTITY_MERGE_JSON_PROMPT_ALL_V3 = """
-Goal-
Analyze a new node against existing nodes to identify potential duplicates in a wearable health knowledge graph.

GUIDELINES FOR COMPARISON:
1. Semantic Analysis:
   - Look beyond exact text matches
   - Consider medical synonyms and related terms
   - Evaluate contextual meaning in healthcare

2. Description Analysis:
   - Identify overlapping concepts
   - Consider complementary information
   - Evaluate scope and specificity

3. Scoring Criteria:
   0.0-0.3: Clearly different concepts
   0.4-0.6: Related but distinct
   0.7-0.8: Highly similar
   0.9-1.0: Virtually identical
   
4. Only return the node if you think it is a duplicate of an existing node.

INPUT FORMAT:
{
    "input_name": "new node name",
    "input_description": "new node description",
    "references": [
        {
            "name": "existing node 1 name",
            "description": "existing node 1 description"
        },
        {
            "name": "existing node 2 name",
            "description": "existing node 2 description"
        }
        ...
    ]
}

OUTPUT FORMAT:
[
    {
        "input_name": "new node name",
        "reference_name": "matched existing node name",
        "similarity_score": <float 0-1>,
        "same_concept": <boolean>,
        "reasoning": "clear explanation of similarity assessment and why the nodes are the same or different"
    }
    ...
]

EXAMPLE :
Input:
{
    "input_name": "steps",
    "input_description": "Total number of steps registered during the day.",
    "references": [
    {
      "name": "Steps taken",
      "description": "Steps Taken refers to the total number of steps registered by a wearable device over a given period, typically a day. It is a key metric for assessing physical activity levels, with higher step counts generally associated with better cardiovascular health, weight management, and overall fitness. Wearable devices track steps using accelerometers or gyroscopes to detect motion and count steps based on movement patterns."
    },
    {
      "name": "Distance traveled",
      "description": "Distance Traveled refers to the total length of the path covered by an object or individual as it moves from one point to another over a given period of time. It is a scalar quantity, meaning it only considers the magnitude of the movement and not the direction. This metric is commonly measured by wearable devices using accelerometers and GPS to track movement and calculate the cumulative distance covered during activities such as walking, running, or cycling."
    },
    {
      "name": "Energy expenditure",
      "description": "Energy Expenditure (EE) refers to the total amount of energy, measured in kilocalories (kcal), that an individual expends over a 24-hour period. It comprises three main components: Basal Metabolic Rate (BMR), the energy required for basic physiological functions at rest; the Thermic Effect of Food (TEF), the energy used to digest and metabolize food; and Physical Activity-Related Energy Expenditure (PAEE), the energy expended during physical activities. EE is a critical metric for understanding caloric balance, weight management, and overall metabolic health. Wearable devices estimate EE using accelerometers, heart rate monitors, and algorithms that combine movement data with physiological parameters."
    },
    {
      "name": "Activity level",
      "description": "Activity Level (AL) quantifies the intensity of physical activity based on Metabolic Equivalent of Task (MET) levels, ranging from sedentary (non-wear) to high-intensity activity. It is a key metric for assessing daily physical activity patterns, energy expenditure, and overall fitness. Wearable devices estimate AL using accelerometers and algorithms that classify movement intensity into categories such as sedentary, lightly active, moderately active, and very active."
    }
  ]
}
Output:
[
    {
      "input_name": "steps",
      "reference_name": "Steps taken",
      "similarity_score": 0.95,
      "same_concept": true,
      "reasoning": "Both nodes refer to the total number of steps registered by a wearable device over a given period, typically a day. The descriptions are nearly identical, with both emphasizing the use of accelerometers or gyroscopes to detect motion and count steps. The terms 'steps' and 'steps taken' are semantically equivalent in this context."
    }
]

"""


ENTITY_MERGE_JSON_PROMPT_ALL_V2 = """
-Goal-
Analyze a new node against existing nodes to identify potential duplicates in a wearable health knowledge graph.

GUIDELINES FOR COMPARISON:
1. Semantic Analysis:
   - Look beyond exact text matches
   - Consider medical synonyms and related terms
   - Evaluate contextual meaning in healthcare
   - Pay special attention to intensity levels and specificity in activity classification
   - Differentiate between related but distinct psychological constructs

2. Description Analysis:
   - Identify overlapping concepts
   - Consider complementary information
   - Evaluate scope and specificity
   - Differentiate between general categories and specific intensity levels
   - Distinguish between different measurement approaches and temporal scopes

3. Scoring Criteria:
   0.0-0.3: Clearly different concepts
   0.4-0.6: Related but distinct
   0.7-0.8: Highly similar
   0.9-1.0: Virtually identical
   
4. Only return the node if you think it is a duplicate of an existing node.

5. Key Differentiators:
   - Different intensity levels (sedentary, light, moderate, vigorous) are DISTINCT concepts
   - General categories vs specific sub-categories are DISTINCT
   - Different measurement units or quantification methods indicate DISTINCT concepts
   - Broader terms vs more specific terms are DISTINCT unless they refer to exactly the same scope
   - Different psychological constructs (e.g., negative affect vs stress) are DISTINCT
   - Different temporal measurements (e.g., asleep duration vs total sleep duration) are DISTINCT if they measure different aspects

INPUT FORMAT:
{
    "input_name": "new node name",
    "input_description": "new node description",
    "references": [
        {
            "name": "existing node 1 name",
            "description": "existing node 1 description"
        },
        {
            "name": "existing node 2 name",
            "description": "existing node 2 description"
        }
        ...
    ]
}

OUTPUT FORMAT:
[
    {
        "input_name": "new node name",
        "reference_name": "matched existing node name",
        "similarity_score": <float 0-1>,
        "same_concept": <boolean>,
        "reasoning": "clear explanation of similarity assessment and why the nodes are the same or different, specifically addressing intensity levels, specificity, psychological constructs, or measurement approaches"
    }
    ...
]

EXAMPLES:
Input:
{
    "input_name": "steps",
    "input_description": "Total number of steps registered during the day.",
    "references": [
    {
      "name": "Steps taken",
      "description": "Steps Taken refers to the total number of steps registered by a wearable device over a given period, typically a day. It is a key metric for assessing physical activity levels, with higher step counts generally associated with better cardiovascular health, weight management, and overall fitness. Wearable devices track steps using accelerometers or gyroscopes to detect motion and count steps based on movement patterns."
    }
  ]
}
Output:
[
    {
      "input_name": "steps",
      "reference_name": "Steps taken",
      "similarity_score": 0.95,
      "same_concept": true,
      "reasoning": "Both nodes refer to the total number of steps registered by a wearable device over a given period, typically a day. The descriptions are nearly identical, with both emphasizing the use of accelerometers or gyroscopes to detect motion and count steps. The terms 'steps' and 'steps taken' are semantically equivalent in this context."
    }
]

Input:
{
    "input_name": "Light active time",
    "input_description": "Time spent in light intensity physical activities, typically measured in minutes per day.",
    "references": [
    {
      "name": "Active time",
      "description": "Total time spent in any physical activity above sedentary level, including light, moderate, and vigorous intensity activities."
    }
  ]
}
Output:
[
    {
      "input_name": "Light active time",
      "reference_name": "Active time",
      "similarity_score": 0.6,
      "same_concept": false,
      "reasoning": "While both concepts relate to physical activity time, they represent different scopes and specificity. 'Light active time' specifically measures time spent in light intensity activities only, whereas 'Active time' is a broader category that includes light, moderate, and vigorous intensity activities combined. These are distinct concepts with different measurement focuses and clinical applications."
    }
]


"""


# ENTITY_MERGE_JSON_PROMPT_ALL_V2 = """
# -Goal-
# Analyze a new node against existing nodes to identify potential duplicates in a wearable health knowledge graph.

# GUIDELINES FOR COMPARISON:
# 1. Semantic Analysis:
#    - Look beyond exact text matches
#    - Consider medical synonyms and related terms
#    - Evaluate contextual meaning in healthcare
#    - Pay special attention to intensity levels and specificity in activity classification
#    - Differentiate between related but distinct psychological constructs

# 2. Description Analysis:
#    - Identify overlapping concepts
#    - Consider complementary information
#    - Evaluate scope and specificity
#    - Differentiate between general categories and specific intensity levels
#    - Distinguish between different measurement approaches and temporal scopes

# 3. Scoring Criteria:
#    0.0-0.3: Clearly different concepts
#    0.4-0.6: Related but distinct
#    0.7-0.8: Highly similar
#    0.9-1.0: Virtually identical
   
# 4. Only return the node if you think it is a duplicate of an existing node.

# 5. Key Differentiators:
#    - Different intensity levels (sedentary, light, moderate, vigorous) are DISTINCT concepts
#    - General categories vs specific sub-categories are DISTINCT
#    - Different measurement units or quantification methods indicate DISTINCT concepts
#    - Broader terms vs more specific terms are DISTINCT unless they refer to exactly the same scope
#    - Different psychological constructs (e.g., negative affect vs stress) are DISTINCT
#    - Different temporal measurements (e.g., asleep duration vs total sleep duration) are DISTINCT if they measure different aspects

# INPUT FORMAT:
# {
#     "input_name": "new node name",
#     "input_description": "new node description",
#     "references": [
#         {
#             "name": "existing node 1 name",
#             "description": "existing node 1 description"
#         },
#         {
#             "name": "existing node 2 name",
#             "description": "existing node 2 description"
#         }
#         ...
#     ]
# }

# OUTPUT FORMAT:
# [
#     {
#         "input_name": "new node name",
#         "reference_name": "matched existing node name",
#         "similarity_score": <float 0-1>,
#         "same_concept": <boolean>,
#         "reasoning": "clear explanation of similarity assessment and why the nodes are the same or different, specifically addressing intensity levels, specificity, psychological constructs, or measurement approaches"
#     }
#     ...
# ]

# EXAMPLES:
# Input:
# {
#     "input_name": "steps",
#     "input_description": "Total number of steps registered during the day.",
#     "references": [
#     {
#       "name": "Steps taken",
#       "description": "Steps Taken refers to the total number of steps registered by a wearable device over a given period, typically a day. It is a key metric for assessing physical activity levels, with higher step counts generally associated with better cardiovascular health, weight management, and overall fitness. Wearable devices track steps using accelerometers or gyroscopes to detect motion and count steps based on movement patterns."
#     }
#   ]
# }
# Output:
# [
#     {
#       "input_name": "steps",
#       "reference_name": "Steps taken",
#       "similarity_score": 0.95,
#       "same_concept": true,
#       "reasoning": "Both nodes refer to the total number of steps registered by a wearable device over a given period, typically a day. The descriptions are nearly identical, with both emphasizing the use of accelerometers or gyroscopes to detect motion and count steps. The terms 'steps' and 'steps taken' are semantically equivalent in this context."
#     }
# ]

# Input:
# {
#     "input_name": "Light active time",
#     "input_description": "Time spent in light intensity physical activities, typically measured in minutes per day.",
#     "references": [
#     {
#       "name": "Active time",
#       "description": "Total time spent in any physical activity above sedentary level, including light, moderate, and vigorous intensity activities."
#     }
#   ]
# }
# Output:
# [
#     {
#       "input_name": "Light active time",
#       "reference_name": "Active time",
#       "similarity_score": 0.6,
#       "same_concept": false,
#       "reasoning": "While both concepts relate to physical activity time, they represent different scopes and specificity. 'Light active time' specifically measures time spent in light intensity activities only, whereas 'Active time' is a broader category that includes light, moderate, and vigorous intensity activities combined. These are distinct concepts with different measurement focuses and clinical applications."
#     }
# ]

# Input:
# {
#     "input_name": "PANAS negative affect",
#     "input_description": "Negative affect score from the Positive and Negative Affect Schedule, measuring emotions like distress, anger, and fear.",
#     "references": [
#     {
#       "name": "Stress",
#       "description": "Psychological and physiological response to challenging or threatening situations, measured through self-report or physiological markers."
#     }
#   ]
# }
# Output:
# [
#     {
#       "input_name": "PANAS negative affect",
#       "reference_name": "Stress",
#       "similarity_score": 0.4,
#       "same_concept": false,
#       "reasoning": "These represent distinct psychological constructs. PANAS negative affect measures specific emotional states like distress, anger, and fear as part of a validated affect scale, while stress refers to a broader physiological and psychological response to challenges. Although related, they measure different aspects of psychological experience and use different assessment methodologies."
#     }
# ]

# Input:
# {
#     "input_name": "Asleep duration",
#     "input_description": "Time spent in confirmed sleep states, excluding periods of wakefulness during the night.",
#     "references": [
#     {
#       "name": "Total sleep duration",
#       "description": "Total time from sleep onset to final awakening, including all sleep stages and brief awakenings."
#     }
#   ]
# }
# Output:
# [
#     {
#       "input_name": "Asleep duration",
#       "reference_name": "Total sleep duration",
#       "similarity_score": 0.5,
#       "same_concept": false,
#       "reasoning": "These are different sleep metrics with distinct measurement approaches. 'Asleep duration' typically refers to actual time spent sleeping excluding wakefulness, while 'total sleep duration' includes the entire sleep period from onset to awakening. They represent different aspects of sleep measurement and have different clinical interpretations."
#     }
# ]

# """



# EXAMPLE:
# Input:
# {
#     "input_name": "heart rate during sleep",
#     "input_description": "Average heart rate measured during sleep periods in bpm",
#     "references": [
#         {
#             "name": "sleep heart rate",
#             "description": "Mean cardiac beats per minute during sleep sessions"
#         }
#     ]
# }

# Output:
# [
#     {
#         "input_name": "heart rate during sleep",
#         "reference_name": "sleep heart rate",
#         "similarity_score": 0.95,
#         "same_concept": true,
#         "reasoning": "Both nodes measure identical physiological parameter (heart rate) in same context (sleep) with matching units (bpm). Terms 'average' and 'mean' are mathematically equivalent."
#     }
# ]

ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT_BATCH = """
-Goal-
Analyze relationships between pairs of wearable health metrics and generate standardized edge representations.
Use provided descriptions, web search results, and your own clinical knowledge to determine meaningful relationships.
be aware that some reference information may not be accurate or related to the entity, so you should be careful to filter out the irrelevant information.
-Input Format-
[
    {
    "id": "<id>",
    "entity_1_name": "<entity_1_name>", 
    "entity_1_description": "<entity_1_description>",
    "entity_2_name": "<entity_2_name>", 
    "entity_2_description": "<entity_2_description>",
    "web_search_results": "<relevant_search_results>",
    }
    ...
]

-Output Format-
[
    {
        "id": str,
        "entity_1_name": str,
        "entity_2_name": str,
        "relationship": {
            "description": str,          # Detailed explanation of the relationship
            "strength": float,           # 0.1 to 1.0
            "confidence": str            # "high", "medium", or "low" based on evidence quality
        },
    },
    ...
]


RELATIONSHIP SCORING:
- Strong (0.7-1.0):
  * Clear scientific evidence
  * Direct causal or strong correlational relationship
  * Well-documented in medical literature

- Moderate (0.3-0.6):
  * Some scientific evidence
  * Indirect or secondary relationship
  * Limited but consistent documentation

- Weak (0.1-0.2):
  * Limited or circumstantial evidence
  * Indirect relationship with multiple variables
  * Inconsistent documentation

- Not Related (<0.1):
  * Exclude from output
  * No meaningful connection
  * No supporting evidence

-Examples-
Example 1:
Input:
[
    {
    "id": "1",
    "entity_1_name": "Heart rate", 
    "entity_1_description": "The number of heartbeats per unit of time, usually expressed as beats per minute."
    "entity_2_name": "Blood pressure", 
    "entity_2_description": "The pressure of the circulating blood against the walls of the blood vessels.",
    "web_search_results": "Elevated heart rate is associated with elevated blood pressure, increased risk for hypertension, and, among hypertensives, increased risk for cardiovascular disease. Despite these important relationships, heart rate is generally not a major consideration in choosing antihypertensive medications.
    Reference 0: Your heart rate increases as you exercise. · Your diastolic blood pressure (the second number in your blood pressure reading) will also rise.
    Reference 1: “For example, if you are dehydrated, bleeding or have a severe infection, blood pressure typically decreases and heart rate increases.” How to ...
    Reference 2: Heart rate and blood pressure measure two different things, both of which affect how hard the heart must work to get blood to the rest of ...
    Reference 3: Due to stimulation from the nervous system, heart rate increases and blood vessels constrict to increase blood pressure. Factors That Affect ...
    Reference 4: Heart rate is the number of times your heart beats per minute. · Blood pressure is the force of blood flowing through your blood vessels.
    Reference 5: Your heart rate can increase without any change occurring in your blood pressure. As your heart beats faster, healthy blood vessels will expand ...
    Reference 6: Heart rate and blood pressure are intimately related. Nerves and hormones constantly monitor and balance the heart rate and blood pressure.
    Reference 7: Elevated heart rate is associated with elevated blood pressure, increased risk for hypertension, and, among hypertensives, increased risk for cardiovascular ...
    Reference 8: While both are important measurements, blood pressure is the more critical of the two. A heart rate that falls outside of the standard for a healthy adult may ...
    "
    },
    {
    "id": "2",
    "entity_1_name": "stress",
    "entity_1_description": "Stress is a physiological and psychological response to external pressures or demands, which are perceived as threatening or challenging. It can result from both positive and negative experiences, such as work deadlines, relationships, or major life changes. Stress triggers a variety of responses in the body, including the release of hormones like cortisol and adrenaline, which prepare the body to either confront or avoid the stressor. While short-term stress can be motivating and adaptive, chronic stress can have negative effects on mental and physical health, leading to conditions like anxiety, depression, and cardiovascular problems.",
    "entity_2_name": "engagement",
    "entity_2_description": "Engagement refers to the emotional and cognitive involvement of an individual in an activity, task, or interaction.",
    "web_search_results": "Psychological distress can negatively affect work engagement. In a study on non-healthcare workers, it was found that there were statistically significant differences between people with and without psychological distress.
    Reference 0: These high levels of stress can cause reduced employee engagement and mental health struggles and this can have a direct impact on employee engagement at work.
    Reference 1: Stressed workers showed lower work engagement and more cognitive complaints, even after adjusting for demographic variables. Negative ...
    Reference 2: There are many ways in which depression, other mental health conditions, and workplace stress contribute to negative occupational outcomes.
    Reference 3: When employees feel mentally well, they are more likely to experience high levels of engagement, as mental health influences core aspects of ...
    Reference 4: Physical activity and coping styles are factors that contribute to health status and to the reduction of stress. The aim of this research ...
    Reference 5: This study explores the relationships between psychosocial factors, work engagement, and mental health among university faculty in Saudi Arabia.
    Reference 6: Engaged employees tend to experience better mental health outcomes, including greater job satisfaction and reduced stress.
    Reference 7: Overall findings suggest that psychological distress has a negative correlation with levels of work engagement.
    Reference 8: Meanwhile, Saleem et al. (2022) demonstrated that stress can moderate the relationship between academic engagement and psychological capital, with high stress ...
    "
    },
    {
    "id": "3",
    "entity_1_name": "blood oxygen",
    "entity_1_description": "Blood oxygen saturation (SpO2) percentage measured via pulse oximetry",
    "entity_2_name": "ambient light",
    "entity_2_description": "Environmental light level in lux",
    "web_search_results": "The greatest difference in pulse oximetry reading between any of the light sources was 0.5%. Repeated-measures analysis of variance yielded a p value of 0.204. Ambient light has no statistically significant effect on pulse oximetry readings.
    Reference 0: OBJECTIVE: Determine whether ambient light affects the accuracy of pulse oximetry readings. DESIGN: Prospective, repeated-measures study.
    Reference 1: Conclusions: Ambient light has no statistically significant effect on pulse oximetry readings. Even had the differences been statistically significant, the ...
    Reference 2: It shows that the relatively stable ambient light intensity has no obvious influences on SPO2 (the average gray value fluctuation is within ± 5). Temperature is ...
    Reference 3: Pulse oximeter manufacturers have designed systems to reject some forms of optical interference, such as ambient light. However, light emanating ...
    Reference 4: The pulsed light of light-emitting diodes can distort pulse oximetry measurements. The stroboscopic effect leads to low saturation values.
    Reference 5: In order to reduce the interference of ambient light on near-infrared blood oxygen sensors (used to measure oxygen saturation of human tissue or oxygen ...
    Reference 6: Light absorption in oxygenated and deoxygenated blood varies appreciably over the visible and near-infrared spectrum.
    Reference 8: The SpO2 level is computed using the red and blue channels, which correspond to the red and infrared light used in contact pulse oximeters. In ...
    "
    }
]
Output: 
[
    {
        "id": "1",
        "entity_1_name": "Heart rate",
        "entity_2_name": "Blood pressure",
        "relationship": {
            "description": "Heart rate and blood pressure are interconnected through the nervous and hormonal systems. Elevated heart rate is associated with elevated blood pressure and increased risk for hypertension and cardiovascular disease. However, heart rate is not always directly proportional to blood pressure, as other physiological factors influence the relationship.",
            "strength": 0.8,
            "confidence": "high"
        }
    },
    {
        "id": "2",
        "entity_1_name": "stress",
        "entity_2_name": "engagement",
        "relationship": {
            "description": "Chronic stress negatively impacts engagement by causing psychological distress, cognitive complaints, and reduced emotional involvement in activities. Conversely, higher engagement levels are associated with better stress coping mechanisms and improved mental health.",
            "strength": 0.6,
            "confidence": "medium"
        }
    }
]


-Real Data-
######################
"""

# 400 tokens per input
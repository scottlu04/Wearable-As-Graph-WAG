
ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT_2 = """
-Goal-
Generate a standardized edge representation between two wearable metric nodes that synthesizes available information(including the description and web search results) with your own clinical knowledge.
be aware that some reference information may not be accurate or related to the entity, so you should be careful to filter out the irrelevant information.
-Input Format-
Entity 1: 
{
    "name": "<entity_1_name>", 
    "description": "<entity_1_description>"
}
Entity 2: 
{
    "name": "<entity_2_name>", 
    "description": "<entity_2_description>"
}
web_search_results: "<relevant_search_results>"

-Output Format-
{
    "relationship": "<detailed_relationship_description>",
    "relationship_strength": <score>
}

-Scoring Guidelines-
Relationship strength should be between 0 and 1:
* 1.0 to 0.7: Strong correlation/relationship
* 0.6 to 0.3: Moderate correlation/relationship
* 0.2 to 0.1: Weak correlation/relationship

-Examples-
Example 1:
Entity 1: 
{
    "name": "Heart rate", 
    "description": "The number of heartbeats per unit of time, usually expressed as beats per minute."
}
Entity 2: 
{
    "name": "Blood pressure", 
    "description": "The pressure of the circulating blood against the walls of the blood vessels."
}
web_search_results: 
"short summary: Elevated heart rate is associated with elevated blood pressure, increased risk for hypertension, and, among hypertensives, increased risk for cardiovascular disease. Despite these important relationships, heart rate is generally not a major consideration in choosing antihypertensive medications.
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
Output: 
{
    "relationship": "Heart rate and blood pressure are closely related physiological metrics. An elevated heart rate is commonly associated with increased blood pressure, as the heart works harder to pump blood, which can constrict blood vessels and elevate pressure. However, this relationship is not absolute, as factors like exercise, dehydration, or vascular health can cause heart rate and blood pressure to change independently. Nervous system stimulation and hormonal regulation also play a role in balancing these two metrics. Chronic elevated heart rate is linked to hypertension and increased cardiovascular disease risk.",
    "relationship_strength": 0.8
}
Example 2:
Entity 1: 
{
    "name": "stress", 
    "description": "Stress is a physiological and psychological response to external pressures or demands, which are perceived as threatening or challenging. It can result from both positive and negative experiences, such as work deadlines, relationships, or major life changes. Stress triggers a variety of responses in the body, including the release of hormones like cortisol and adrenaline, which prepare the body to either confront or avoid the stressor. While short-term stress can be motivating and adaptive, chronic stress can have negative effects on mental and physical health, leading to conditions like anxiety, depression, and cardiovascular problems."
}
Entity 2: 
{
    "name": "engagement", 
    "description": "Engagement refers to the emotional and cognitive involvement of an individual in an activity, task, or interaction. "
}
web_search_results: 
"Psychological distress can negatively affect work engagement. In a study on non-healthcare workers, it was found that there were statistically significant differences between people with and without psychological distress.
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
Output: 
{
    "relationship": "Stress and engagement are inversely related in many contexts, particularly in work and academic settings. High levels of stress are associated with reduced emotional and cognitive involvement in tasks, as individuals struggle to focus and maintain motivation under pressure. Several studies highlight that psychological distress, which includes stress, negatively affects engagement levels. Conversely, individuals with lower stress levels or better mental health often report higher engagement, indicating that well-being is critical for maintaining focus and enthusiasm in activities. However, some evidence suggests that moderate stress can motivate individuals to achieve goals, though this effect is context-dependent and often short-lived.",
    "relationship_strength": 0.7
}

Example 3:
Entity 1: 
{
    "name": "blood oxygen", 
    "description": "Blood oxygen saturation (SpO2) percentage measured via pulse oximetry"
}
Entity 2: 
{
    "name": "ambient light", 
    "description": "Environmental light level in lux"
}
web_search_results: "
The greatest difference in pulse oximetry reading between any of the light sources was 0.5%. Repeated-measures analysis of variance yielded a p value of 0.204. Ambient light has no statistically significant effect on pulse oximetry readings.
Reference 0: OBJECTIVE: Determine whether ambient light affects the accuracy of pulse oximetry readings. DESIGN: Prospective, repeated-measures study.
Reference 1: Conclusions: Ambient light has no statistically significant effect on pulse oximetry readings. Even had the differences been statistically significant, the ...
Reference 2: It shows that the relatively stable ambient light intensity has no obvious influences on SPO2 (the average gray value fluctuation is within ± 5). Temperature is ...
Reference 3: Pulse oximetry is based on the principle that O 2 Hb absorbs more near-IR light than HHb, and HHb absorbs more red light than O 2 Hb.
Reference 4: Pulse oximeter manufacturers have designed systems to reject some forms of optical interference, such as ambient light. However, light emanating ...
Reference 5: The pulsed light of light-emitting diodes can distort pulse oximetry measurements. The stroboscopic effect leads to low saturation values.
Reference 6: In order to reduce the interference of ambient light on near-infrared blood oxygen sensors (used to measure oxygen saturation of human tissue or oxygen ...
Reference 7: Light absorption in oxygenated and deoxygenated blood varies appreciably over the visible and near-infrared spectrum.
Reference 8: The SpO2 level is computed using the red and blue channels, which correspond to the red and infrared light used in contact pulse oximeters. In ...
"
Output: "unclear"


-Real Data-
######################
"""


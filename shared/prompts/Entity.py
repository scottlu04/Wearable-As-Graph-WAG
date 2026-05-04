


ENTITY_GENERATION_JSON_PROMPT_2 = """
-Goal-
Generate a standardized node representation for a wearable sensor measurement that synthesizes available information with your own clinical knowledge.
be aware that some reference data may not be related to the entity, so you should be careful to filter out the irrelevant information but the provided data is important and should always be used.
-Input Format-
{
    "entity_name": "<measurement_name>",
    
    // source: Device/sensor documentation
    "provided_description": "<provided_description>",
    "provided_range": "<provided_range>", 

    // source: web search
    "web_description": "<web_search_results>",  
    "value_range": "<ranges>",                 
    "recommendations": "<guidelines>"   

    
    // source: UMLS
    "umls_description": "<umls_definition>",   
    
}

-Output Format-
{
    "name": "<standardized_name>",
    "description": "<comprehensive_description>",
    "range": "<range_info_or_None>",
    "recommendations": "<guidelines_or_None>"
}

-Guidelines-
1. Name:
   - Use standardized medical terminology
   - Keep concise but clear
   - Include common abbreviation if applicable

2. Description (Required):
   - What is being measured
   - How it's measured
   - Clinical significance
   - Relationship to health outcomes

3. Range (if applicable):
   - Normal ranges for different demographics
   - Units of measurement
   - Alert thresholds
   - Output "None" if not applicable(e.g. gesture recognized does not have a range)

4. Recommendations (if applicable):
   - Evidence-based
   - Actionable
   - Context-aware
   - Output "None" if not applicable (e.g. there is not such a recommendation for improvement for the entity, like gesture)

-Examples-
Example 1:
Input:
{
  "entity_name": "Energy expenditure",
  "provided_description": "Energy consumption caused by the physical activity of the day.",
  "provided_range": "range: None unit: kcal",
  "web_description": "As people pursue activities at multiple locations, trips are produced between successive activity locations. Patterns formed by trips over a period, such as a day, are called 'travel patterns.\nReference 0: Pattern means two or more acts occurring over a period of time, however short, evidencing a continuity of purpose. Digital Cross Connect System or ...\nReference 1: A tourist's travel pattern is a combination of the destinations they choose to visit, the order they choose to visit them in, and why they make ...\nReference 2: Travel Patterns show comparisons of the number of Facebook location history users moving across long distances, like cross-country air or train travel.\nReference 3: These land use patterns have a significant effect on travel patterns, including number of trips, trip length and time, and the choice of mode.\nReference 4: NYMTC's Travel Patterns summarizes, annually and by each quarter of the year, the average weekday ridership on buses, rail, subways, and ferries.\nReference 5: State the main characteristics of the following travel segments: regular business travel; meetings, conventions and congresses; incentive travel ...\nReference 6: Travel patterns subsume all relevant information to describe the move- ments of people. They include information about when, how, where, and ...\nReference 7: points by land use category, Trip distance, Trip duration, Start and end time, Complete routing information for each trip (network links and transit routes) ...\nReference 8: Travel patterns represent models of the spatial-temporal relationship among the destinations on a particular trip, which is essential information for planning ...\n",
  "umls_description": "",
  "value_range": "\nReference 0: A travel pattern refers to the classification of daily travel behaviors based on factors such as the number of trips and total travel time.\nReference 1: Long-distance trips are journeys of more than 50 miles from home to the furthest destination.\nReference 2: New MIT research confirms people visit places more frequently when they have to travel shorter distances to get there.\nReference 3: Travel Patterns show comparisons of the number of Facebook location history users moving across long distances, like cross-country air or train travel. Our ...\nReference 4: As noted in the section above, the majority of the commute times are within the 21-30 minute range. Shorter commute times are reported for areas closer to ...\nReference 5: This paper presents a methodology to extract passenger archetypes based on the long-distance travel mobility patterns observed through MND. This methodology ...\nReference 6: Patterns of travel \u00b7 Distance. Distance is a combination of the time and money it costs to travel from origin to destination. \u00b7 International ...\nReference 7: Introduction. This report uses data from the 2022 National Household Travel Survey (NHTS) to examine the daily travel patterns of American ...\nReference 8: distance travel. Page 99. UNDERSTANDING REGIONAL TRAVEL PATTERNS. Page | 86. Table 141: Trip Distance of SR 166 East Traffic. Trip. Distance. (miles). Eastbound.\nReference 9: We suggest a positioning range of 300 meters\u2013400 meters is appropriate to obtain sufficient location accuracy. Table 4. The percentages of trajectory ...\n",
  "recommendations": "\nReference 0: Almost every peer review pointed out the importance of integrating land development patterns into the travel demand models.\nReference 1: This paper carries out a literature review to identify the key elements of mobility apps that foster more sustainable travelers' choices.\nReference 2: This paper evaluates the usefulness of morphological characteristics of the urban street network of the city of S\u00e3o Carlos (Brazil) by analyzing university ...\nReference 3: The main recommendations emanating from the research include monitoring domestic tourism trends and patterns to inform marketing and product development ...\nReference 4: We offer the following recommendations to enhance the depth, relevance and accuracy of NTTO's travel and tourism data programs. The current ...\nReference 5: Some of the most promising innovations could improve travel time reliability, that is, increase predictability for the same trips compared day-to-day.\nReference 6: The accuracy of travel studies is usually improved by integrating travel surveys or travel activity diaries with GPS data. GPS surveys are implemented to ...\nReference 7: Improving Our Understanding of How Highway Congestion and Pricing Affect Travel Demand (2012) Chapter: Chapter 6 - Conclusions and Recommendations for Future ...\nReference 8: We developed a personalized service-trajectory correlation that could recommend the most appropriate services to users.\nReference 9: Giving today's travelers what they need and want \u00b7 Know customer segments inside and out \u00b7 Help travelers share their journeys \u00b7 Recognize younger ...\n"
}
Output:
{
    "name": "Energy Expenditure (EE)",
    "description": "Energy Expenditure (EE) refers to the total amount of energy, measured in kilocalories (kcal), that an individual expends through physical activity and metabolic processes over a given period, typically a day. It is a key metric for understanding caloric balance, weight management, and overall physical activity levels. Wearable devices estimate EE using accelerometers, heart rate monitors, and algorithms that combine movement data with physiological parameters.",
    "range": "Energy expenditure varies widely based on factors such as age, sex, weight, and activity level. For example, sedentary adults may expend 1,800-2,400 kcal/day, while highly active individuals may expend 3,000 kcal/day or more. Units: kilocalories (kcal).",
    "recommendations": "To optimize energy expenditure, engage in regular physical activity, such as aerobic exercises, strength training, and daily movement. Monitor caloric intake to align with energy expenditure goals, and consider using wearable devices to track activity levels and adjust behaviors accordingly."
}


Example 2:
Input:
{
  "entity_name": "Sleep efficiency",
  "provided_description": "",
  "provided_range": "",
  "web_description": "Sleep efficiency is the percentage of time spent asleep while in bed. It is calculated by dividing the amount of time spent asleep (in minutes) by the total amount of time in bed (in minutes). A normal sleep efficiency is considered to be 85% or higher.\nReference 0: Sleep efficiency (SE), commonly defined as the ratio of total sleep time (TST) to time in bed (TIB), plays a central role in insomnia research and practice.\nReference 1: Sleep efficiency (SE) is the ratio between the time a person spends asleep, and the total time dedicated to sleep It is given as a percentage.\nReference 2: The definition of sleep efficiency is the percentage of time we spend asleep while in bed. Time spent in bed not trying to sleep\u2014while reading, ...\nReference 3: Healthy sleep quality goes beyond quantity. It refers to how well you sleep and is measured by how fast you fall asleep among other factors.\nReference 4: Sleep efficiency is another important parameter that refers to percentage of total time in bed actually spent in sleep.\nReference 5: Introduction. Sleep efficiency, defined as the ratio of total sleep time to time in bed, decreases clearly and significantly with age [1\u20134]. It ...\nReference 6: When sleep professionals talk about sleep efficiency, they are generally referring to the percentage of time a person spends asleep at night.\nReference 7: Sleep Efficiency \u2013 Sleep efficiency refers to the percentage of time a person sleeps, in relation to the amount of time a person spends in bed. The percentage ...\nReference 8: Sleep Efficiency is a simple but powerful metric--the percentage of the time you spend in bed that you are actually asleep.\n",
  "umls_description": "A relative measurement (percentage) of the time spent asleep (N1 sleep + N2 sleep + N3 sleep + REM sleep) to the total time spent in bed.\n",
  "value_range": "Sleep efficiency (SE) is the ratio between the time a person spends asleep, and the total time dedicated to sleep (i.e. both sleeping and attempting to fall asleep or fall back asleep). It is given as a percentage. SE of 80% or more is considered normal/healthy with most young healthy adults displaying SE above 90%.\nReference 0: The proposed formula for SE would be: SE = TST / DSE (\u00d7 100). DSE can be easily calculated using standard sleep diary entries along with one item from the ...\nReference 1: A normal sleep efficiency is considered to be 85% or higher. ... Related web pages: Diagnosis, classification, symptoms, and causes of hypersomnias \u00b7 Glossary.\nReference 2: The total ESS score is based on a scale of 0 to 24, with a score equal to and above 16 considered to be very sleepy and warrants further investigation. Total ...\nReference 3: You can calculate your sleep efficiency by dividing the time you're asleep by the total time in bed. So, if you sleep for six out of eight hours ...\nReference 4: Normal sleep efficiency is considered to be 80% or greater. For example, if a person spends 8 hours in bed (from 10 p.m. to 6 a.m), at least 6.4 hours or more ...\nReference 5: The graph above shows that the average Sleep Efficiency on WHOOP is 94.4%, meaning that for a typical 8-hour time in bed, the average WHOOP user ...\nReference 6: Sleep efficiency above 85% is considered good; below this is typically considered insomnia. Keeping it above 90% is ideal. That said, good sleep ...\nReference 7: A healthy sleep efficiency is generally considered to be around 85% or higher. Here's how this range breaks down: 80-89%: This is often considered the lower ...\nReference 8: Moreover, knowing that the average sleep efficiency in this population ranges from 76% to 81% [4, 40], the intention was to address a ...\n",
  "recommendations": "Avoid prolonged use of light-emitting screens just before bedtime. Consider using room-darkening shades, earplugs, a fan or other devices to create an environment that suits your needs. Doing calming activities before bedtime, such as taking a bath or using relaxation techniques, might promote better sleep.\nReference 0: Day 1: Wake up at the same time each day. \u00b7 Day 2: Remove electronics from the bedroom. \u00b7 Day 3: Lock the pets out of the bedroom. \u00b7 Day 4: ...\nReference 1: You can begin by establishing a consistent bedtime routine within 30 to 60 minutes of the time you want to sleep that prepares your body to wind down.\nReference 2: 1. Invest in a Better Mattress and Bedding \u00b7 2. Block Out Light \u00b7 3. Minimize Noise \u00b7 4. Set the Thermostat to 65 to 68 Degrees Fahrenheit \u00b7 5. Get ...\nReference 3: Making changes to your diet and lifestyle may help improve your sleep quality. Certain supplements and natural remedies could also be beneficial.\nReference 4: Keep a consistent routine. \u00b7 Get morning light. \u00b7 Exercise during the day helps improve your sleep quality at night, reduces stress, and improves ...\nReference 5: Good sleep is essential for our health and emotional well-being. \u00b7 Getting enough sleep and good sleep quality are essential for healthy sleep.\nReference 6: Examples of a bedtime routine include the following: turning off bright lights and disengaging from stimulating activities, meditating, reading, listening to ...\nReference 7: Refrain from exercise at least 4 hours before bedtime. Regular exercise is recommended to help you sleep well, but the timing of the workout is ...\nReference 8: Irregular sleep schedule: Going to bed and waking up at the same time each day can improve sleep. \u00b7 Sleeping environment: The bedroom should be ...\n"
}
Output:
{
    "name": "Sleep Efficiency (SE)",
    "description": "Sleep Efficiency (SE) is the percentage of time spent asleep compared to the total time in bed, calculated as the ratio of Total Sleep Time (TST) to Time in Bed (TIB) multiplied by 100. It is a critical metric in assessing sleep quality and identifying disorders like insomnia, with higher values indicating more efficient sleep. Wearable sensors estimate SE using accelerometers or heart rate variability to determine sleep and wake states.",
    "range": "Normal SE is 80%-90% for healthy adults, with values below 80% indicating potential sleep disturbances. Units: percentage (%).",
    "recommendations": "Maintain a consistent sleep schedule, avoid screen time before bed, optimize the sleeping environment with reduced light and noise, and engage in relaxing activities before bedtime."
}

Example 3:
Input:
{
  "entity_name": "Gesture",
  "provided_description": "",
  "provided_range": "",
  "web_description": "1. : a movement usually of the body or limbs that expresses or emphasizes an idea, sentiment, or attitude. raised his hand overhead in a gesture of triumph. 2. : the use of motions of the limbs or body as a means of expression.\nReference 0: a movement of the body, hands, arms, or head to express an idea or feeling: He made a rude gesture to the crowd after his tennis match.\nReference 1: Gestures include movement of the hands, face, or other parts of the body. Gestures differ from physical non-verbal communication that does not communicate ...\nReference 2: a movement of the hands or body, but it's also a movement that has some meaning, intention, or emotion behind it.\nReference 3: a movement of your body (especially of your hands and arms) that shows or emphasizes an idea or a feeling. Specific gestures can indicate particular moods.\nReference 4: A gesture is a movement of the hand, arms, or other body part that is intended to indicate or emphasize something, often when speaking.\nReference 5: A gesture is a movement that you make with a part of your body, especially your hands, to express emotion or information.\nReference 6: A meaningful body movement, usually of the hand or head, though the term can include facial expression and expressive movements of the whole body.\nReference 7: A gesture may also be considered an act or a remark made as a formality or as a sign of intention or attitude. [10] This definition applies to the acts such as ...\n",
  "umls_description": "Pohyb \u010d\u00e1st\u00ed t\u011bla za \u00fa\u010delem komunikace.\nMovimento de uma parte do corpo visando a comunica\u00e7\u00e3o.\nMovimiento de una parte del cuerpo con el prop\u00f3sito de comunicarse.\nMovement of a part of the body for the purpose of communication.\nDet \u00e5 bevege deler av kroppen med det form\u00e5l \u00e5 kommunisere.\nR\u00f6relse av kroppsdelar i kommunikationssyfte. \"Kroppsspr\u00e5k\".\n",
  "value_range": "\nReference 0: In this work, we address the Ultra-Range Gesture Recognition (URGR) problem by aiming for a recognition distance of up to 25 m and in the context of HRI. We ...\nReference 1: In this work, we address the Ultra-Range Gesture Recognition (URGR) problem by aiming for a recognition distance of up to 25 meters and in the ...\nReference 2: The hand gestures are collected at discrete distance levels from 1 meter to 7 meters, where most of the hand gestures are small and at low resolution.\nReference 3: Gestures include movement of the hands, face, or other parts of the body. Gestures differ from physical non-verbal communication that does not communicate ...\nReference 4: At a time in development when children are limited in the words they know and use, gesture offers a way to extend their communicative range.\nReference 5: The main purpose of this study was to develop an accurate hand gesture recognition system that is capable of error-free auto-landmark localization of any ...\nReference 6: This GR10-30 gesture sensor is capable of recognizing 12 hand gestures: move up, down, left, right, forward, backward, rotate clockwise and counterclockwise.\nReference 7: Gesture is one of the important channels of man-machine interface. In the paper, methods to recognize hand gestures using range images are proposed.\nReference 8: This paper proposes a real time hand gesture recognition system. The approach uses a range camera to capture the depth data. After some pre- processing ...\n",
  "recommendations": "\nReference 0: Most trainers incorporate some combination of peer review and video recording. Video is particularly useful for spotting uncontrolled gestures and eye movement.\nReference 1: I recommend breaking the body down into a rectangle for the chest/torso, oval for the hips, circle for the head, and triangles for hands. Then ...\nReference 2: We explore several different design and methodological considerations for gesture research specific to persons with aphasia.\nReference 3: 5 Gesture Drawing Tips \u00b7 1. Keep your lines fluid. \u00b7 2. Forget the details. \u00b7 3. Draw from life. \u00b7 4. Leave the eraser behind. \u00b7 5. Experiment with ...\nReference 4: I think what is holding back your gesture is your dedication to shapes. The page on the middle right that has the two figures on one sheet, to me, has the best ...\nReference 5: Gestures are a crucial prelinguistic skills. In this post you'll learn 3 tips for encouraging gestures in your Early Intervention sesions.\nReference 6: Use hand movements purposefully to emphasize points, but avoid excessive or distracting gestures. Pay attention to your posture, stand tall, and ...\nReference 7: Keep the lines loose and dynamic to reflect the energy of the pose. Practice regularly to improve your ability to capture gestures effectively.\nReference 8: Keep solid eye contact, while also consulting your notes often as to make the words keep flowing. Make hand gestures. Some people prefer holding ...\nReference 9: http://publicspeakingpower.com/better-public-speaking-gestures/ - View the blog post for full written transcription as well as downloadable ...\n"
}

Output:
{
    "name": "Gesture",
    "description": "A gesture is a purposeful movement of the hands, arms, face, or other body parts to convey information, emotions, or intentions, often accompanying verbal communication. Gestures play a critical role in human interaction, aiding expression and enhancing understanding, especially in non-verbal or prelinguistic contexts. They can also be utilized in gesture recognition systems for human-computer interaction.",
    "range": "",
    "recommendations": ""
}
-Real Data-
######################
"""
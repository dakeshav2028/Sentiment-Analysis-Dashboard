# Qualitative Error Analysis

**Model Analyzed:** Fallback Pretrained (lxyuan/distilbert-base-multilingual-cased-sentiments-student)
**Test Set Accuracy:** 0.5580
**Total Misclassified Reviews in Test Set:** 221 out of 500

Below are 8 analyzed examples of misclassifications, showing the text, star rating, ground truth, and predicted sentiment, along with a diagnostic comment.

### Example 1
- **Review Text:** "Great concept.... Just not good enough for larger, heavier, phones.

Great concept, but doesn't work on heavier larger phones like the Note 8. The phone is simply too heavy with a basic style case.The slightest bump will cause the phone to slip off the mount and fall to the floor, potentially damaging your phone. Unfortunately this resulted in me returning this, as it simply did not work as well as I had hoped for my Note 8."
- **Star Rating:** 3 Stars
- **Ground Truth Label:** Neutral
- **Predicted Label:** Negative
- **Error Category:** Mixed Sentiment
- **Analysis:** The review contains both positive and negative aspects (e.g., 'good product but bad packaging'). The model struggled to weigh the opposing sentiment clauses correctly.

### Example 2
- **Review Text:** "Seems to work

I see improvement with my Great Pyrenees but it’s hard to get him to eat these. Update - he won't eat these anymore. What can you do."
- **Star Rating:** 3 Stars
- **Ground Truth Label:** Neutral
- **Predicted Label:** Negative
- **Error Category:** Mixed Sentiment
- **Analysis:** The review contains both positive and negative aspects (e.g., 'good product but bad packaging'). The model struggled to weigh the opposing sentiment clauses correctly.

### Example 3
- **Review Text:** "For narrow feet only

Very cute shoe but is for a very narrow foot."
- **Star Rating:** 3 Stars
- **Ground Truth Label:** Neutral
- **Predicted Label:** Positive
- **Error Category:** Mixed Sentiment
- **Analysis:** The review contains both positive and negative aspects (e.g., 'good product but bad packaging'). The model struggled to weigh the opposing sentiment clauses correctly.

### Example 4
- **Review Text:** "Cute, but multiple bags needed.

The colors were cute and girly. Perfect for our 1 year old daughter. The balls are not very sturdy. She is able to squeeze them, I make sure not to crush them if I get in the pit with her. 1 bag is definitely not enough. We bought 3 bags of 200 balls and maybe could have even gotten away with one more."
- **Star Rating:** 3 Stars
- **Ground Truth Label:** Neutral
- **Predicted Label:** Positive
- **Error Category:** Mixed Sentiment
- **Analysis:** The review contains both positive and negative aspects (e.g., 'good product but bad packaging'). The model struggled to weigh the opposing sentiment clauses correctly.

### Example 5
- **Review Text:** "Not big enough for smart key

Not big enough for smart key"
- **Star Rating:** 2 Stars
- **Ground Truth Label:** Negative
- **Predicted Label:** Neutral
- **Error Category:** Subtle / Sarcastic Tone
- **Analysis:** The sentiment is expressed through sarcasm or subtle context that simple word associations cannot easily capture.

### Example 6
- **Review Text:** "Very heavy... certainly not for backpacking

This thing isn't waterproof and it's very heavy.. probably not the best choice for normal fishing... If your typical fish is less than #100 then you should go much lighter. -2 Stars for weight"
- **Star Rating:** 3 Stars
- **Ground Truth Label:** Neutral
- **Predicted Label:** Negative
- **Error Category:** Ambiguous 3-Star Rating
- **Analysis:** 3-star reviews are often highly neutral or moderately mixed. The star rating mapping is inherently noisy here, as the reviewer might sound slightly positive or negative, causing a mismatch with the 'Neutral' ground truth.

### Example 7
- **Review Text:** "Coverage was fair

Coverage was fair but came off easily once applied. Also, has a kind of Minty type smell to it that I didn't care for"
- **Star Rating:** 3 Stars
- **Ground Truth Label:** Neutral
- **Predicted Label:** Positive
- **Error Category:** Mixed Sentiment
- **Analysis:** The review contains both positive and negative aspects (e.g., 'good product but bad packaging'). The model struggled to weigh the opposing sentiment clauses correctly.

### Example 8
- **Review Text:** "HORRIBLE

HORRIBLE DONT GET IT IT IS THE WORSE THING EVER IT MIGHT LOOK GOOD BUT IT IS NO"
- **Star Rating:** 1 Stars
- **Ground Truth Label:** Negative
- **Predicted Label:** Positive
- **Error Category:** Mixed Sentiment
- **Analysis:** The review contains both positive and negative aspects (e.g., 'good product but bad packaging'). The model struggled to weigh the opposing sentiment clauses correctly.


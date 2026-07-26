# Qualitative Error Analysis

**Model Analyzed:** Fine-tuned BERT-Tiny
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
- **Review Text:** "Seams rip easily.

I use this for my daughter’s clothes. She’s 3, so her clothes aren’t big or heavy. We live in CA, so no heavy winter gear or jackets, just short sleeve regular t shirts and shorts, but somehow one of the seams of the boxes ripped. Look at other reviews, this is a common problem."
- **Star Rating:** 2 Stars
- **Ground Truth Label:** Negative
- **Predicted Label:** Neutral
- **Error Category:** Mixed Sentiment
- **Analysis:** The review contains both positive and negative aspects (e.g., 'good product but bad packaging'). The model struggled to weigh the opposing sentiment clauses correctly.

### Example 3
- **Review Text:** "Two Stars

Looked cheap, had a hard time using side buttons. Returned it."
- **Star Rating:** 2 Stars
- **Ground Truth Label:** Negative
- **Predicted Label:** Neutral
- **Error Category:** Mixed Sentiment
- **Analysis:** The review contains both positive and negative aspects (e.g., 'good product but bad packaging'). The model struggled to weigh the opposing sentiment clauses correctly.

### Example 4
- **Review Text:** "Its trash, but you get what you pay for.

Spend an extra ten or twenty dollars and get something that will last."
- **Star Rating:** 1 Stars
- **Ground Truth Label:** Negative
- **Predicted Label:** Neutral
- **Error Category:** Mixed Sentiment
- **Analysis:** The review contains both positive and negative aspects (e.g., 'good product but bad packaging'). The model struggled to weigh the opposing sentiment clauses correctly.

### Example 5
- **Review Text:** "Not big enough for smart key

Not big enough for smart key"
- **Star Rating:** 2 Stars
- **Ground Truth Label:** Negative
- **Predicted Label:** Positive
- **Error Category:** Subtle / Sarcastic Tone
- **Analysis:** The sentiment is expressed through sarcasm or subtle context that simple word associations cannot easily capture.

### Example 6
- **Review Text:** "Works great

Exactly as advertised. I installed it in my garage on the ceiling. I mostly use it to run a table saw and air compressor during projects. I have had no issues so far. It retracts well and is heavy duty enough that I Know it will handle anything I will be using it for."
- **Star Rating:** 5 Stars
- **Ground Truth Label:** Positive
- **Predicted Label:** Neutral
- **Error Category:** Subtle / Sarcastic Tone
- **Analysis:** The sentiment is expressed through sarcasm or subtle context that simple word associations cannot easily capture.

### Example 7
- **Review Text:** "Big??? Cheaper more comfortable and their big NOT THESE

Large but tightly made, feels small in shower, too small, too tight to add soap and clean with...put the rest away.. never to be used again."
- **Star Rating:** 1 Stars
- **Ground Truth Label:** Negative
- **Predicted Label:** Neutral
- **Error Category:** Mixed Sentiment
- **Analysis:** The review contains both positive and negative aspects (e.g., 'good product but bad packaging'). The model struggled to weigh the opposing sentiment clauses correctly.

### Example 8
- **Review Text:** "Four Stars

Nice, but a bit heavy."
- **Star Rating:** 4 Stars
- **Ground Truth Label:** Positive
- **Predicted Label:** Neutral
- **Error Category:** Mixed Sentiment
- **Analysis:** The review contains both positive and negative aspects (e.g., 'good product but bad packaging'). The model struggled to weigh the opposing sentiment clauses correctly.


# 🛒 Instacart Sequential Mining — 2-Minute Video Script

**Target Duration:** 2 minutes (120 seconds ± 2 sec)  
**Word Count:** ~295 words  
**Style:** Screen capture of you talking over the 5 slides.

---

### [0:00 – 0:07] SLIDE 1: Title Slide

"Hi, I'm Khussal Pradhan. My project is Instacart Market Basket Analysis — using FP-Growth and PrefixSpan to uncover temporal patterns in grocery shopping."

### [0:07 – 0:30] SLIDE 2: The Grocery Blind Spot

"This dataset contains 3.4 million orders across 200,000 users and 50,000 products. But here's the problem: current recommendation systems only look at what's in your cart *right now*. Traditional algorithms like FP-Growth find co-purchases but completely ignore *when* you buy things. They have no temporal memory — and that blind spot costs millions in wasted inventory and missed revenue every year."

### [0:30 – 0:55] SLIDE 3: Predicting the Future Cart

"My project fixes this with two complementary approaches: FP-Growth to find items bought *together* in one basket, and PrefixSpan — a sequential pattern mining algorithm — to find baskets bought *in sequence* over a user's entire order history. The key design choice: each user's history is structured as a chronological sequence of canonicalized basket tuples — preserving order boundaries so PrefixSpan can mine real basket-to-basket trajectories."

### [0:55 – 1:22] SLIDE 4: The 7 & 14-Day Rhythm

"The EDA revealed that shopping behavior is deeply cyclical — distinct reorder spikes at 7, 14, 21, and 30 days. With over 99.9% sparsity in the data, only extremely low support thresholds work. At our optimal threshold of half a percent, FP-Growth discovered 344 significant association rules. We also compared early-week versus late-week shoppers and found distinctly different purchasing patterns between the two segments."

### [1:22 – 2:00] SLIDE 5: What We Found & Where It Goes

"So what did sequential mining add? PrefixSpan discovered over 10,000 sequential patterns. The key result: 133 item pairs are bought in *sequence* across orders but *never* together in the same basket. For example, Organic Raspberries followed by Organic Zucchini — never co-purchased, but consistently bought one after the other. FP-Growth is completely blind to these temporal dependencies. The takeaway: if we can predict what a user needs next week, we can pre-stock warehouses, push perfectly timed coupons, and build recommendation engines that finally understand time. Thank you."

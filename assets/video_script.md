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

"My project fixes this by analyzing a 5,000-user reproducible sample with two complementary approaches: FP-Growth to find items bought *together* in one basket, and PrefixSpan to find baskets bought *in sequence* over a user's entire order history. The key design choice: each user's history is structured as a chronological sequence of canonicalized basket tuples — preserving order boundaries so PrefixSpan can mine real basket-to-basket trajectories."

### [0:55 – 1:22] SLIDE 4: The 7 & 14-Day Rhythm

"The EDA revealed that shopping behavior is highly cyclical — a dominant reorder peak at 7 days, with visible periodicity at 14, 21, and 30 days. With over 99.9% sparsity in the data, only extremely low support thresholds work. At our optimal threshold of half a percent, FP-Growth discovered 344 significant association rules. We also compared early-week versus late-week shoppers and found distinctly different purchasing patterns between the two segments."

### [1:22 – 2:00] SLIDE 5: What We Found & Where It Goes

"So what did sequential mining add? PrefixSpan discovered over 10,000 sequential patterns, including multi-step trajectories spanning three or more consecutive shopping trips. An honest apples-to-apples comparison showed that at the pair level, most sequential items also co-occur somewhere. But PrefixSpan's real value is capturing the *temporal ordering* — the multi-step trajectory of *what comes after what* — which co-occurrence analysis structurally cannot represent. The takeaway: if we can model these purchasing trajectories, we can predict next week's basket, pre-stock warehouses, and build recommendation engines that finally understand time. Thank you."

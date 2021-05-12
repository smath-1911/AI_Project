# AI_Project

The objective of this project is to build a Movie Recommendation system with  Reinforcement Learning to understand its implementation and try to incorporate a hybrid movie filtering technique.

A traditional recommender systems have been modeled with two paradigms, collaborative filtering and content-based systems.
In collaborative filtering-based methods, the recommendation is built over the “user-item interaction matrix”, which are records of users’ past interaction with the items. The underlying concept for collaborative filter-based methods is to detect similar users and their interest based on their proximity. A collaborative filtering algorithm can be built on the following methods: memory based, and model based. In the memory-based method, for a new user, the most similar user is identified, and their most liked content is recommended. In the memory-based method, there is no concept of variance or bias as the error cannot be quantified. In the model-based method, a generative model is built on top of the user-item interaction matrix and the model is then used to predict new users. In this variant of modeling, model bias and variance are observed.
In content-based recommendation systems, apart from the user-item interaction, the user information and preferences are also taken into account, and other details related to content like popularity, description, or purchase history, etc. The user features and content features are fed into a model, which works like a traditional machine learning model with error optimization. As this model, contains more descriptive information related to the content, it tends to have high bias, but lowest variance compared to other modeling methods.

Here we are also studying the model based on greedy approach or linear Upper Confidence Bound approach. When choosing the action with the highest estimated value, you are said to be “choosing greedily” and the actions with the maximum values, of which there may be more than one, are know as the “greedy actions”. Greedy Algorithm wants to choose the action with the best estimate, yet provides no way to form these estimates. It purely exploits the information available, but does none of the exploration required to generate this information.

Exploration allows the agent to improve its knowledge about each action. Hopefully, leading to a long-term benefit. Exploitation allows the agent to choose the greedy action to try to get the most reward for short-term benefit. A pure greedy action selection can lead to sub-optimal behaviour. A dilemma occurs between exploration and exploitation because an agent can not choose to both explore and exploit at the same time. Hence, we use the Upper Confidence Bound algorithm to solve the exploration-exploitation dilemma.
UCB action selection uses uncertainty in the action-value estimates for balancing exploration and exploitation. Since there is inherent uncertainty in the accuracy of the action-value estimates when we use a sampled set of rewards thus UCB uses uncertainty in the estimates to drive exploration.

![image](https://user-images.githubusercontent.com/65208476/117999367-c6bd2580-b312-11eb-88c2-a835fbf86f22.png)




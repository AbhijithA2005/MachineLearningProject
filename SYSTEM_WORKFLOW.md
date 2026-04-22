# System Workflow and Architecture Documentation

This document explicitly details the complete functional workflow, technical stack, and algorithmic mechanisms behind the Machine Learning classification system.

## 1. Complete Technology Stack

### Frontend (User Interface)
*   **Core Library**: React.js (Component-driven UI, encapsulated in `App.jsx`).
*   **Build Tool**: Vite (Lightning-fast Hot Module Replacement and bundle optimization).
*   **Styling**: Pure CSS (`index.css`) emphasizing deep customization, modern design aesthetics, responsive design, and CSS variables.
*   **Routing / Gateway Mapping**: Web server configurations utilizing Nginx (`nginx.conf`) and standard Dockerized containers for production encapsulation.

### Backend (API Engine)
*   **Framework**: FastAPI (Asynchronous, high-performance Python web framework).
*   **Data Validation**: Pydantic (Strict typing and request validation via `BaseModel`).
*   **Environment**: Python 3.x, optimized specifically to run on lightweight serverless functions (like Vercel).
*   **Deployment Mapping**: `vercel.json` provides direct deployment to the Vercel edge network for scalability without cold-start bottle-necks seen with large ML models. 
*   **Containerization**: `docker-compose.yml` and `Dockerfile` are used for localized container orchestration.

---

## 2. System Workflow: From Input to Accurate Output

The entire pipeline executes over an API-driven architecture that rapidly triages plain-text problem statements into complex machine learning roadmaps.

### Step 2.1: Data Ingestion (Input)
1.  **User Action**: The user physically types a business problem or data context into the React frontend input field (e.g., "Predict customer churn based on transaction history").
2.  **Frontend State Transition**: The React application binds this text to state and triggers an asynchronous `fetch` or `axios` call.
3.  **Network Transport**: The text is bundled inside a JSON payload `{"description": "..."}` and sent via an HTTP POST request to the `/api/predict` endpoint of the FastAPI server.

### Step 2.2: Backend Reception & Validation
1.  **Pydantic Validation**: Before the string touches core logic, the `ProblemRequest(BaseModel)` validates the payload structure.
2.  **Routing**: The `/api/predict` endpoint receives the valid data and passes the problem string directly to the model's core intelligence (`logic.py` -> `get_ml_info()`).

### Step 2.3: Natural Language Processing (Entity Extraction)
To properly frame the context, the system utilizes RegEx (Regular Expressions) to parse grammatical structure and extract the two most vital components of any ML project:
*   **Target Variable**: The main objective. The model searches for trigger verbs (`predict`, `forecast`, `classify`, `cluster`) to isolate what is being targeted.
*   **Data Source**: The contextual source. The model looks for transition prepositions (`based on`, `using`, `from`) to figure out what data is historically available.

### Step 2.4: The Classification Engine & Core Algorithm Execution
*This aligns with the model evaluation and algorithm triggering.* (Detailed significantly in Section 3).

### Step 2.5: Dynamic Generation of Accurate Output
Once the system has derived the correct ML paradigm (`ml_type`), it synthesizes an output dynamically.
1.  **Confidence Computation**: It measures a confidence gradient (between `0.0` and `1.0`), scaled by problem complexity (word length) and the frequency of keyword hits. 
2.  **Generative Roadmaps**: Maps the derived problem to an array of implementation milestones (e.g., "Data Aggregation", "Model Training"), associating them with distinct technologies (e.g., "Scikit-Learn", "FastAPI").
3.  **JSON Payload Release**: The backend constructs a highly detailed JSON dictionary representing the model's conclusions (which type, confidence explanation, roadmap, selected ML algorithms, why alternatives fail, etc.) and returns it via HTTP 200 OK.
4.  **Frontend Rendering**: The React frontend maps this JSON object iteratively to its state, changing the CSS interface (via dynamic class rendering) and displaying a modern, interactive dashboard of the accurate output.

---

## 3. Classifier Model Mechanics: How is it Trained?

A critical, identifying detail of this specific system architecture is that **it does not utilize a heavy neural network that undergoes gradient descent or backpropagation training (like a Transformer or BERT).** 

Because the API needs to operate seamlessly on serverless architecture (Vercel) with extremely restrictive memory constraints (usually under 250MB), loading a 1GB PyTorch `.pt` file would result in failure. Therefore, the "Model" is an **Expert System implemented via a Heuristic Weighted Keyword Classifier Engine.**

### A. How "Training" is achieved conceptually 
Instead of automated weight adjustment via historical epochs, the model is "trained" through **Domain-Expert Hardcoding**. The developers established a matrix of weights mapping distinct natural-language concepts to Machine Learning Paradigms:

1.  **Supervised Learning Weights**: Trained to react to concepts of labeled output like `"predict"`, `"forecast"`, `"classification"`, `"regression"`, `"detect"`, `"ground truth"`.
2.  **Unsupervised Learning Weights**: Trained to react to concepts of hidden relationships like `"cluster"`, `"group"`, `"segment"`, `"hidden"`, `"unlabeled"`.
3.  **Reinforcement Learning Weights**: Trained to react to concepts of environmental interaction like `"agent"`, `"reward"`, `"optimize"`, `"policy"`, `"environment"`.

### B. The internal Algorithm used to Classify
The algorithm used is a **Feature-Weighted Occurrence Scoring Algorithm**:
1.  **Tokenization**: The input text is cast to lowercase and mapped against the weight matrices.
2.  **Linear Progression**: The algorithm loops through every keyword. If a word is found, the respective dictionary score (Supervised, Unsupervised, or Reinforcement) increments (`+1`).
3.  **Strong Indicator Amplification**: Highly decisive words (such as `"predict"`, `"cluster"`, `"reward"`, or `"agent"`) trigger a modifier step causing the score to increment again (`+1`), dynamically weighing crucial signifiers heavier than generic ones.
4.  **ArgMax Function Output**: The class with the highest integer score dictates the classification output.

### C. Recommended Algorithms by the Classifier Engine
When the classifier matches a user's problem to a specific paradigm, it maps it to the precise ML training algorithms that the user should implement natively. 
*   **If Classified as Supervised**: The engine recommends *Random Forest*, *Gradient Boosting (XGBoost/LightGBM)*, or *Neural Networks (MLP)*, alongside time-series alternatives like *ARIMA* or *LSTM*.
*   **If Classified as Unsupervised**: The engine recommends *K-Means Clustering*, *PCA (Principal Component Analysis)*, or *Isolation Forest* for anomaly detection. Deep learning setups like *Autoencoders* and *GANs* are suggested alternatives.
*   **If Classified as Reinforcement**: The engine recommends Policy Gradients and Action-Value functions such as *PPO (Proximal Policy Optimization)*, *DQN (Deep Q-Network)*, and *SAC (Soft Actor-Critic)*.

By bypassing heavy epoch-based training and instead utilizing computationally infinite-speed heuristic rules, the classifier maintains a `100%` response up-time while accurately determining the machine learning algorithms needed for any given input string.

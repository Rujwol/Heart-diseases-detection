# Heart-diseases-detection

The dataset used in this project is the Cleveland Heart Disease dataset taken from the UCI repository. The dataset consists of 303 individual's data. In the actual dataset we had 76 features but for our study we chose only 14 important ones. They are:
1. Age
2. Sex
3. Chest-pain type
4. Resting Blood Pressure
5. Serum Cholesterol
6. Fasting Blood Sugar
7. Resting ECG
8. Max heart rate achieved
9. Exercise induced angina
10. ST depression induced by exercise relative to rest
11.  Peak exercise ST segment
12. Number of major vessels (0–3) colored by fluoroscopy
13. Thal 
14. Diagnosis of heart disease
URL: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
 
METHODS AND ALGORITHMS USED
Naïve Bayes
Naive Bayes is a simple but an effective classification technique which is based on the Bayes Theorem.  It assumes  independence among  predictors,  i.e.,  the  attributes  or  features  should  be  not correlated to one another or should not,  in anyway, be related to each other. Even if there is dependency, still all these features or attributes independently contribute to the probability and that is why it is called Naïve.
 
Support Vector Machine
Support  Vector  Machine  is  an  extremely  popular  supervised machine learning technique(having a predefined  target variable) which  can  be  used  as  a  classifier  as  well  as    predictor.  For classification, it finds a hyper-plane in the feature space that differentiates between the classes. An SVM model represents the training data points as points in the feature space, mapped in such a way that points belonging to separate classes are segregated by a margin as wide as possible. The test data points are then mapped into that same space and are classified based on which side of the margin they fall.
 
Decision Tree
Decision trees are supervised learning algorithms. This technique is mostly used in classification problems.  It performs effortlessly with continuous and categorical attributes.  This algorithm divides the population into two or more similar sets based on the most significant predictors. Decision Tree algorithm, first calculates the entropy of each and every attribute.  Then the dataset is  split with  the  help  of  the variables  or  predictors  with  maximum information  gain  or  minimum  entropy.  These two steps are performed recursively with the remaining attributes.
 
Random Forest
Random Forest is also a popularly supervised machine learning algorithm. This technique can be used for both regression and classification tasks but generally performs better in classification tasks. As the name suggests, Random Forest technique considers multiple decision trees before giving an output. So, it is basically an ensemble of decision trees.  This  technique  is  based  on  the belief  that  more trees  would  converge  to  the  right decision.  For  classification,  it  uses  a  voting  system  and  then decides the class whereas in regression it takes the meaning of all the outputs  of  each  of  the decision  trees.  It works well with large datasets with high dimensionality.

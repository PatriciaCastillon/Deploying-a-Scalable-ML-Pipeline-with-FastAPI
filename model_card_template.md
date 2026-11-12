# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model was developed by Patricia Castillon as part of a project on deploying a scalable machine learning pipeline with FastAPI. It is a Random Forest classifier built with scikit-learn version 1.5.1 (`RandomForestClassifier`) using `random_state=42` and otherwise default hyperparameters. The model predicts whether a person's annual income exceeds $50,000 based on demographic and employment attributes. The trained model and the fitted one-hot encoder are saved as pickle files in the `model/` folder. The code is written in Python 3.10.22, and the repository uses GitHub Actions to run pytest and flake8 on every push.

## Intended Use

The model is intended for educational purposes, to demonstrate an end-to-end machine learning workflow that includes training, evaluation, slice-based performance analysis, and deployment behind an API. It takes a person's demographic and employment attributes and returns a prediction of either `<=50K` or `>50K`. It should not be used to make real decisions about individuals, such as decisions about lending, hiring, housing, or insurance.

## Training Data

The model was trained on the UCI Census Income (Adult) dataset, which was extracted from the 1994 U.S. Census database. The file `census.csv` contains [32,561] rows and [15] columns. The target column is `salary`, and about 76% of records fall in the `<=50K` class, so the classes are imbalanced. The features include numeric columns such as age, education-num, capital-gain, capital-loss, and hours-per-week, as well as eight categorical features: workclass, education, marital-status, occupation, relationship, race, sex, and native-country. The data was split 80/20 into training and test sets using a stratified split with `random_state=42`, so both sets keep the same class balance. The training set contains [26,048] rows. The categorical features were one-hot encoded and the label was binarized, with both transformations fitted on the training data only.

## Evaluation Data

The evaluation data is the 20% held-out test set from the same census file, which contains 6,513 rows. It was processed with the encoder and label binarizer fitted on the training data, so no information from the test set leaked into training. Because the test set comes from the same source and time period as the training data, it measures performance on similar data rather than on new populations.

## Metrics


The model was evaluated with precision, recall, and the F1 score (F-beta with beta equal to 1). Precision is the share of people predicted to earn more than $50K who actually do. Recall is the share of people who actually earn more than $50K that the model correctly identifies. F1 is the harmonic mean of the two. These metrics were chosen over plain accuracy because the classes are imbalanced, and a model that always predicted `<=50K` would score about 76% accuracy without learning anything useful.

On the test set, the model achieved a precision of 0.7327, a recall of 0.6397, and an F1 score of 0.6830. This means that about 73% of the people the model flags as high earners truly are, but it misses roughly 36% of the people who actually earn more than $50K.

Performance was also computed on slices of the test data, with one set of metrics for every unique value of each categorical feature. The full results are in `slice_output.txt`. Performance is similar across sex, with an F1 score of 0.6847 for Female and 0.6827 for Male, although the Female slice had higher precision (0.7638 versus 0.7274) and slightly lower recall (0.6204 versus 0.6432). Across race, the F1 score was 0.6855 for White (5,533 rows), 0.6708 for Black (662 rows), 0.6337 for Asian-Pac-Islander (200 rows), and 0.5333 for Amer-Indian-Eskimo (73 rows), so the smaller groups tend to score lower. Performance varies much more by education: F1 was highest for Masters (0.8533), Doctorate (0.8889), and Prof-school (0.8778), and lowest for 12th grade (0.2857), 9th grade (0.2500), and 7th-8th grade (0.0000), where the model found none of the high earners. Occupation shows a similar spread, with F1 ranging from 0.7967 for Prof-specialty down to 0.2667 for Other-service. Relationship also matters: Husband and Wife slices scored 0.6983 and 0.7421, while Own-child scored only 0.3158. Slices with very few rows, such as native-country values like France (2 rows) or Jamaica (21 rows), produced extreme scores of 0.0000 or 1.0000 that are not statistically meaningful.

## Ethical Considerations

The dataset contains sensitive attributes, including race, sex, and native country, and it reflects income patterns in the United States in 1994. The model may learn and reproduce historical inequalities, such as pay gaps between groups, and treat them as legitimate signals. The slice results show that error rates differ across groups, and in particular that recall for the high-income class is lower for smaller racial groups and for people with less formal education, which means the model may be less reliable for some populations than for others. The data also describes real individuals' demographic and financial characteristics, so any use of similar data should respect privacy and applicable regulations. The model should not be used in settings where a wrong prediction could cause real harm to a person.

## Caveats and Recommendations

The data is more than thirty years old, so the relationship between demographics and income has likely changed, and income thresholds are no longer comparable after inflation. The class imbalance leads to weaker recall on the `>50K` class, as the overall recall of 0.6397 shows. No hyperparameter tuning was performed, so performance could probably be improved. Slices with very few examples, such as rare native countries, produce unreliable metrics, and the `zero_division=1` setting can make some of them look perfect when they are not. Before any real-world use, the model should be retrained on current data, tuned, and audited for fairness across demographic groups. It would also help to add confidence thresholds or calibration and to monitor performance over time.
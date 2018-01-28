# Identify Fraud from Enron Email Project

Why this Project?

This project will teach you the end-to-end process of investigating data through a machine learning lens.

It will teach how to extract and identify useful features that best represent your data, a few of the most commonly used machine learning algorithms today, and how to evaluate the performance of your machine learning algorithms.

What will I learn?

By the end of the project, will be able to:

1. Deal with an imperfect, real-world dataset

2. Validate a machine learning result using test data

3. Evaluate a machine learning result using quantitative metrics

4. Create, select and transform features

5. Compare the performance of machine learning algorithms

6. Tune machine learning algorithms for maximum performance

7. Communicate your machine learning algorithm results clearly

Project Overview:

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.

In this project, I will play detective, and put my new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. This project has combined this data with a hand-generated list of persons of interest in the fraud case.

The goal of this project is to identify a person of interest. Which means an individual who was indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity. POI identifier based on financial and email data made public as a result of the Enron scandal. Machine learning is an excellent tool for this kind of classification task as it can use patterns discovered from labeled data to infer the classes of new observations.

The dataset had significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. It could contain useful information for identifying POIs. Insider Salary, Bonus, Long Term Incentive, Deferred Income, Total Payments, Exercised Stock Options, Total Stock Value and number of to/from emails messages are very usefull dataset information for the investigation.

The features in the data fall into three major types, namely financial features, email features and POI labels.

A) financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

B) email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

C) POI label: [‘poi’] (boolean, represented as integer)

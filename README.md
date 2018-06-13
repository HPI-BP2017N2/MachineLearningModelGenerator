# MachineLearningModelGenerator [![Build Status](https://travis-ci.org/HPI-BP2017N2/MachineLearningModelGenerator.svg?branch=master)](https://travis-ci.org/HPI-BP2017N2/MachineLearningModelGenerator)
The model generator is a microservice component, which generates and stores classifiers for various machine learning approaches.
It is written in Java and uses the Spring framework.

## Getting started
### Prerequisites
1. Cache  
 The model generator requests specific offers of idealo from [the cache](https://github.com/HPI-BP2017N2/Cache).
2. MongoDB  
 The model generator uses  MongoDB for loading the results of the matching process. Those data are used for training the classifiers. It is expected that those information is separated into multiple collections (one for every shop, named by the corresponding shop ID).
3. File System   
 The model generator saves he different classifiers used by [the matcher](https://github.com/HPI-BP2017N2/Matcher) on disk in a folder named "models".  
  3.1. One file is named model.json. It contains a serialized classifier used for deciding whether a parsed offer and an idealo offer match or not.  
  3.2. One file is named category.json. It contains one serialized neural network for classifying the category of a parsed offer.  
  3.3. One file is named brand.json. It contains one serialized neural network for classifying the brand of a parsed offer.  
  
### Configuration
#### Environment variables
- MLMG_PORT: The port that should be used by the model generator
- MONGO_IP: The IP of the MongoDB instance
- MONGO_PORT: The port of the MongoDB instance
- MONGO_MLMG_USER: The username to access the MongoDB
- MONGO_MLMG_PW: The password to access the MongoDB
- CACHE_IP: The URI of the cache microservice

#### Component properties
- matchesPerShop: Base amount of offers that should be in training data per shop.
  This value will be undercut when maximum amount of matches would be exceeded.
- maximumMatchesForLearning: Maximum size of training data
- trainingSetPercentage: Percentage of training set.
- labelThreshold: The minimum probability to classify the category and the brand of a parsed offer

## How it works
1. The model generator (MLMG) receives a request to generate a specific classifier (neural network for brand/category classification or model for matching) or all three models together.
2. If not already loaded, MLMG will create testing and training set (if all three classifiers should be trained, this will always perform).  
 2.1. The MLMG gets results matched with EAN (correct matches) from all shops and divides them randomly into training and testing set.  
 2.2. For generation of the model, 50% of matching results are used for match class, 50% are shuffled for not-match class.
3. The MLMG trains the requested classifier(s).
4. If model was requested, the MLMG evaluates all trained models on the training set and chooses the best one.
5. The classifier(s) are stored in file system.

## Future work
- do not store models in file system
- change training and testing set generation to get non-matches with a high similarity
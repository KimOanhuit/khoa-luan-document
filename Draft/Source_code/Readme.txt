*********************************************************************************************************************************
*	1 - Material data	-	Data crawled from  website								*
*	2 - Cooked Data 	-	Data after preprocessing, official data, per file: each line is a document content	*
*********************************************************************************************************************************
*																*
*	How to run 														*
*********************************************************************************************************************************
*	1 - Run make_model.py	*	Create model for classifier, this is a Doc2Vector model 				*
*					- mincount	-	ignore all words with total frequency lower than this		*
*					- window	-	is the maximum distance between the predicted word 		*
*								and context words used for prediction within a document		*
*					- size		-	is the dimensionality of the feature vectors			*
*					- epoch		-	number of iterations over the corpus				*
*					- workers	-	use this many worker threads to train the model 		*
*																*
*		If you heavn't prepaired data yet	answer 0 the chief, he should cook for you a great meal 		*
*		If you have prepaired data 		answer 1 to enjoy your meal :)						*
*																*		*		The result model will be saved and ready to be used								*
*																*
*	2 - Run classifier.py 	*	Applying the classifier algorithm and evaluate its performance				*
*					- K-Fold	-	Cross validation						*
*																*
*																*
*********************************************************************************************************************************

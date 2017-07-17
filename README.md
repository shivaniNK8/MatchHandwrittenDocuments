# MatchHandwrittenDocuments
Generates similarity score between 0 and 1 for given handwritten documents.

Given a set of digitized handwritten documents, the system outputs a similarity score for the two documents. Similarity was calculated using word distribution, thus taking into account paraphrasing. Various word classifiers were trained experimented with, eg. SVM with SIFT features (Bag of words), CNN (Convolutional Neural Networks). 

This method avoids the conversion of text to OCR, which is not reliable for most type of documents. The assigned similarity score would be decided irrespective of factors like word form variation, order in which words appear in different documents, format of the document and paraphrasing. We have used word spotting which is a known technique to retrieve text from images.

The project has three modules:
• Training the CNN using the word images dataset.
• Given two input document images, we get the constituent words from the document images using segmentation of the image.
• Next, trained CNN is used for predicting the class of each segmented word and calculating similarity score.

Applications: Plagiarism detection, automatic scoring of answersheets.

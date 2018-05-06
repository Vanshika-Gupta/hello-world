#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Decision Tree Classification Example.
"""
from __future__ import print_function

from pyspark import SparkContext

# $example on$
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import MulticlassMetrics
import time
import pandas

# $example off$

#class CV_data(object):
    #pass


if __name__ == "__main__":

    sc = SparkContext(appName="PythonDecisionTreeClassificationExample")

    # $example on$
    # Load and parse the data file into an RDD of LabeledPoint.
    data = MLUtils.loadLibSVMFile(sc, 'feature_libsvm.txt')

    time_start = time.time()

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3], seed= 100)

    # Train a DecisionTree model.
    #  Empty categoricalFeaturesInfo indicates all features are continuous.
    model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={2:7, 6:2},
                                         impurity='gini', maxDepth=7, maxBins=32)

    time_end = time.time()
    time_dt = (time_end - time_start)
    print("DT takes %d s" % (time_dt))


    # Evaluate model on test instances and compute test error
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(
        lambda lp: lp[0] != lp[1]).count() / float(testData.count())
    print('Test Error = ' + str(testErr))
    print ('accuracy is : %g' % (1-testErr))
    print('Learned classification tree model:')
    print(model.toDebugString())


    # Save and load model
    #model.save(sc, "myDecisionTreeClassificationModel")
    #sameModel = DecisionTreeModel.load(sc, "target/tmp/myDecisionTreeClassificationModel")

    # INSTANTIATE METRICS OBJECT
    metrics = BinaryClassificationMetrics(labelsAndPredictions)

    # AREA UNDER PRECISION-RECALL CURVE
    print("Area under PR = %s" % metrics.areaUnderPR)

    # AREA UNDER ROC CURVE
    print("Area under ROC = %s" % metrics.areaUnderROC)
    metrics = MulticlassMetrics(labelsAndPredictions)

    # OVERALL STATISTICS
    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()
    confusion_matrix = metrics.confusionMatrix().toArray()


    print("Summary Stats")
    print("Precision = %s" % precision)
    print("Recall = %s" % recall)
    print("F1 Score = %s" % f1Score)
    print ('Confusion Matrix = %s' % confusion_matrix)


    # $example off$


# PREDICT ON TEST DATA WITH BEST/FINAL MODEL
#predictionAndLabels = oneHotTESTbinary.map(lambda lp: (float(logitBest.predict(lp.features)), lp.label))




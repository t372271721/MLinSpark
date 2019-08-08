// Databricks notebook source
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


// Load the data stored in LIBSVM format as a DataFrame.
val trainingData = spark.read.format("libsvm").load("/FileStore/tables/jui07e9f1498208108508/poker_train-fef7a.libsvm")
val testData = spark.read.format("libsvm").load("/FileStore/tables/jui07e9f1498208108508/poker_test-2412b.libsvm")

// Split the data into training and test sets (30% held out for testing)
//val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)

// Train a NaiveBayes model.
val model = new NaiveBayes()
  .fit(trainingData)

// Select example rows to display.
val predictions = model.transform(testData)
predictions.show(10)

// Select (prediction, true label) and compute test error
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " +(1.0 - accuracy))

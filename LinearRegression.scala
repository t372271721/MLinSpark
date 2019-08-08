// Databricks notebook source
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator

// Load training data
val data = spark.read.format("libsvm")
  .load("/FileStore/tables/pu9qim6x1498342511430/YearPredictionMSD.libsvm")

val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

val lr = new LinearRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)

// Fit the model
val lrModel = lr.fit(trainingData)

// Print the coefficients and intercept for linear regression
//println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

// Summarize the model over the training set and print out some metrics
val trainingSummary = lrModel.summary
//println(s"numIterations: ${trainingSummary.totalIterations}")
//println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
//trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")

val predictions = lrModel.transform(testData)
predictions.show(10)
val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)
